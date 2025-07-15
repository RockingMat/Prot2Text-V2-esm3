"""
Stage 2 - instruction tuning training script for ESM-LLAMA protein description 
generation on Esm2LlamaInstructForCausalLM model. 

With LoRA. 

DistributedDataParallel training script implemented from scratch. 

The script currently supports gradient accumulation, AutoMixedPrecision, 
inter-epoch evaluation. 

The script currently does not support save/load pretrained, gradient checkpointing 
or generation under FSDP. 

reference for AMP: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html 

* The script is designed for multi-GPU parallelism on single node.
* On the cluster, print(...) will go to stdout and tqdm(...) will go to stderr.
"""

import argparse
from datetime import datetime
import os
from typing import Any, Dict, Union

from peft import get_peft_model, LoraConfig
from peft.peft_model import PeftModel
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.configuration_esmc_llm import ESMCLLMConfig, ModalityAdapterConfig
from models.modeling_esmc_llm import ESMCambrianLLMInstructForCausalLM

from dataset import Prot2TextLightDataset, Prot2TextLightCollater
import scripts.utils_argparse as utils_argparse


argParser = argparse.ArgumentParser()

argParser.add_argument("--esm_model_name", type=str, default="esmc_600m", help="ESM C model name (esmc_300m, esmc_600m)")
argParser.add_argument("--esm_path", type=str, help="[DEPRECATED] ESM path - now using ESM C models")
argParser.add_argument("--llama_path", type=str)
# argParser.add_argument("--root_dataset_dir", type=str)
argParser.add_argument("--root_csv_dir", type=str)
argParser.add_argument("--save_checkpoint_dir", type=str)
argParser.add_argument("--load_model_checkpoint_path", type=str, default="")
argParser.add_argument("--load_adapter_checkpoint_dir", type=str, default="")
argParser.add_argument("--load_optimizer_scheduler_checkpoint_path", type=str, default="")

argParser.add_argument("--torch_dtype", type=utils_argparse.str2dtype)
argParser.add_argument("--batch_size_per_device", type=int)
argParser.add_argument("--num_epochs", type=int)
argParser.add_argument("--save_every_epochs", type=int)
argParser.add_argument("--gradient_accumulation_steps", type=int)
argParser.add_argument("--learning_rate", type=float)
argParser.add_argument("--gradient_clipping", type=float, default=None)
argParser.add_argument("--scheduler_gamma", type=float)
argParser.add_argument("--random_seed", type=int)
argParser.add_argument("--fix_modality_adapter", type=utils_argparse.str2bool)
argParser.add_argument("--lora_rank", type=int)
argParser.add_argument("--adapter_intermediate_dim", type=int, default=2048, help="Intermediate dimension for the modality adapter")

argParser.add_argument("--include_text_fields", type=utils_argparse.str2bool)
argParser.add_argument("--name_dropout", type=float)
argParser.add_argument("--taxonomy_dropout", type=float)

argParser.add_argument("--train_split", type=str)
argParser.add_argument("--eval_split", type=str)
argParser.add_argument("--debug_trim_train_split", type=int, default=None)
argParser.add_argument("--debug_trim_eval_split", type=int, default=None)


def create_model_config(args: Dict[str, Any]) -> ESMCLLMConfig:
    """Create and validate model configuration."""
    config = ESMCLLMConfig(
        llm_model_name = args.get("llm_model_name", "qwen/Qwen-14B"),
        adapter_config = ModalityAdapterConfig(
            input_dim       = 0,  # set dynamically from ESM encoder hidden size
            intermediate_dim= args.get("adapter_intermediate_dim", 2048),
            output_dim      = 0,  # set dynamically from LLM decoder hidden size
        )
    )
    
    # Validate intermediate dimension
    if config.adapter_config.intermediate_dim <= 0:
        raise ValueError(f"adapter_intermediate_dim must be positive, got {config.adapter_config.intermediate_dim}")
    
    return config


def load_model(args: Dict[str, Any]) -> PeftModel:
    """
    Standard API for different models. Used in both `train` and `generate`.
    Loads the Prot2Text pipeline class (ESM Cambrian encoder + adapter +
    any HF CausalLM, defaulting to Qwen-14B), then wraps it with LoRA.

    Returned model is on CPU under the default dtype.
    """
    # 1) Build our config (specify LLM and adapter size)
    config = create_model_config(args)

    # 2) (Optional) Preload a custom LLM decoder on CPU
    llm_decoder = None
    llm_path = args.get("llm_decoder_path") or args.get("llama_path")
    if llm_path:
        llm_decoder = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype = args["torch_dtype"],
            device_map  = "cpu",
            trust_remote_code = True,
        )

    # 3) Instantiate the full model (ESM Cambrian + adapter + LLM)
    model = ESMCambrianLLMInstructForCausalLM(
        config      = config,
        llm_decoder = llm_decoder,  # if None, class loads from config.llm_model_name
    )
    # 4) (Optional) Load state dict for the base pipeline
    ckpt = args.get("load_model_checkpoint_path")
    if ckpt:
        print(f"Loading base checkpoint from {ckpt}")
        state_dict = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    # 5) Freeze base encoder & decoder weights
    model.esm_encoder.requires_grad_(False)
    model.llm_decoder.requires_grad_(False)

    # 6) Wrap with LoRA
    adapter_ckpt = args.get("load_adapter_checkpoint_dir")
    if adapter_ckpt:
        print(f"Loading LoRA adapter from {adapter_ckpt}")
        model = PeftModel.from_pretrained(
            model,
            adapter_ckpt,
            is_trainable=True
        )
    else:
        print("Initializing new LoRA adapter")
        lora_config = LoraConfig(
            r               = args["lora_rank"],
            lora_alpha      = args["lora_rank"] * 2,
            lora_dropout    = args.get("lora_dropout", 0.1),
            bias            = "none",
            init_lora_weights = True,
            target_modules  = [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ],
            modules_to_save = (
                ["adapter.fc1", "adapter.fc2"]
                if not args.get("fix_modality_adapter", False)
                else None
            ),
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model


def teacher_forcing_forward_pass(
        rank: int,
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        data_batch: Dict[str, Any],
) -> torch.Tensor:  # loss
    """
    Simplified forward pass for ESM Cambrian model.
    Uses raw protein sequences from the dataset.
    """
    return model(
        input_ids=data_batch["input_ids"].to(rank),
        attention_mask=data_batch["attention_mask"].to(rank),
        labels=data_batch["labels"].to(rank),
        protein_sequences=data_batch["protein_sequences"],  # Raw sequences
        use_cache=False,
        return_dict=False,
    )[0]


def setup(rank: int, world_size: int):
    """
    Initialize processes for distributed training before first epoch. 
    Fetch from job script or launcher to set the IP address and the port of the 
    master node. 
    """
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '9901')
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    """End processes for distributed training after last epoch"""
    dist.destroy_process_group()


def train_epoch(
        rank: int,
        current_epoch: int,
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        dataloader: DataLoader,
        optimizer: Optimizer,
        args: Dict[str, Any]
):
    """Iterate over all batches for one epoch in training with teacher forcing"""
    model.train()
    ddp_loss = torch.zeros(2).to(rank)  
        # [0] for acc. loss and [1] for num. of seen batches
    ddp_gradnorm = torch.zeros(2).to(rank)  
        # [0] for acc. gradnorm and [1] for num. of passed steps
    optimizer.zero_grad()  # erase accumulated gradients from last epoch

    t = tqdm(iter(dataloader))
    for batch_idx, data_batch in enumerate(t):
        # with autocast, logits will be in AUTOCAST_DTYPE 
        # but loss will be re-casted to torch.float32
        # and model weights will stay in torch.float32
        loss = teacher_forcing_forward_pass(
            rank=rank, 
            model=model, 
            data_batch=data_batch, 
        )

        # rescale loss for consistency with different gradient accumulation steps
        loss = loss / args["gradient_accumulation_steps"]

        # summary current batch
        t.set_postfix({
            "mode": "train",
            "epoch": f"{current_epoch}/{args['num_epochs']}",
            "batch_loss": loss.item() * args["gradient_accumulation_steps"],
            "device": f"rank:{rank}"
        })
        ddp_loss[0] += loss.item() * args["gradient_accumulation_steps"]
        ddp_loss[1] += 1  # the loss is the weighted mean of the output of every batch

        # scale the loss up by a large factor to prevent them from becoming too small
        # then accumulate the scaled grads
        loss.backward()  
            # backward out of autocast, but still uses same dtype as for forward

        # update weights by loss if accumulation step is reached
        if (batch_idx + 1) % args["gradient_accumulation_steps"] == 0: 
            gradnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    float("inf") 
                    if args["gradient_clipping"] is None 
                    else args["gradient_clipping"]
                )
            )
            ddp_gradnorm[0] += gradnorm
            ddp_gradnorm[1] += 1

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # summary current epoch
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            f"[epoch={current_epoch}/{args['num_epochs']}, "
            f"train_loss={ddp_loss[0] / ddp_loss[1]}, "
            f"epoch_lr={optimizer.param_groups[0]['lr']}, "
            f"epoch_gradnorm={ddp_gradnorm[0] / ddp_gradnorm[1]}]"
        )
        # NaN detection
        if ddp_loss[0] != ddp_loss[0]:
            raise ValueError(
                "NaN detected in the training loss of the epoch, training interrupted."
            )


def eval_epoch(
        rank: int,
        current_epoch: int, 
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        dataloader: DataLoader,
        args: Dict[str, Any]
):
    """Iterate over all batches in evaluation with teacher forcing"""
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)  
        # [0] for acc. loss and [1] for num. of seen batches

    t = tqdm(iter(dataloader))
    for data_batch in t:
        with torch.no_grad():
            loss = teacher_forcing_forward_pass(
                rank=rank,
                model=model,
                data_batch=data_batch,
            )

            t.set_postfix({
                "mode": "eval",
                "epoch": f"{current_epoch}/{args['num_epochs']}",
                "batch_loss": loss.item(),
                "device": f"rank:{rank}"
            })
            ddp_loss[0] += loss.item()
            ddp_loss[1] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            f"[epoch={current_epoch}/{args['num_epochs']}, "
            f"eval_loss={ddp_loss[0] / ddp_loss[1]}]"
        )


def train_on_device(
        rank: int,
        world_size: int,
        args: Dict[str, Any]
):
    """
    Training and evaluation process for each device, including epochs of training 
    with teacher forcing. 
    """
    setup(rank, world_size)

    # prepare datasets and dataloaders
    # Use ESM C for protein encoding - no separate tokenizer needed
    llama_tokenizer = AutoTokenizer.from_pretrained(
        args["llama_path"], 
        pad_token='<|im_end|>',  # Use <|im_end|> for Qwen3 fine-tuning
        trust_remote_code=True  # Required for Qwen3
    )

    train_dataset = Prot2TextLightDataset(
        csv_path=os.path.join(args["root_csv_dir"], f"{args['train_split']}.csv"),
    )
    if args["debug_trim_train_split"]:
        train_dataset.data = train_dataset.data[:args["debug_trim_train_split"]]
    train_sampler = DistributedSampler(
        train_dataset, 
        rank=rank, 
        num_replicas=world_size, 
        shuffle=True
        )
    
    train_collater = Prot2TextLightCollater(
        sequence_tokenizer=None,  # No tokenization needed for ESM C
        description_tokenizer=llama_tokenizer,
        mode="train", 
        include_text_fields=args["include_text_fields"],
        name_dropout=args["name_dropout"],
        taxonomy_dropout=args["taxonomy_dropout"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size_per_device"],
        sampler=train_sampler,
        collate_fn=train_collater,
        num_workers=4,  # parallel CPU cores used for data loading
        pin_memory=True,  # enable page-locked memory allocation for faster data transfer to GPUs
        shuffle=False,  # avoid shuffling twice with DistributedSampler
        drop_last=True,  # avoid incomplete batch at the end
    )
    print(f"Train dataset loaded on rank:{rank}")

    eval_dataset = Prot2TextLightDataset(
        csv_path=os.path.join(args["root_csv_dir"], f"{args['eval_split']}.csv"),
    )
    if args["debug_trim_eval_split"]:
        eval_dataset.data = eval_dataset.data[:args["debug_trim_eval_split"]]
    eval_sampler = DistributedSampler(
        eval_dataset, 
        rank=rank, 
        num_replicas=world_size, 
        shuffle=False
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args["batch_size_per_device"],
        sampler=eval_sampler,
        collate_fn=train_collater,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    print(f"Eval dataset loaded on rank:{rank}")

    torch.cuda.set_device(rank)

    model = load_model(args=args)
    model = model.to(rank)

    model = DistributedDataParallel(
        model, 
        # find_unused_parameters=True  # suppress error for unused parameters in wrapped model
    )
    print(f"DDP model loaded on rank:{rank}")

    # initialization of the optimizer after wrapping the model
    optimizer = Adam(model.parameters(), lr=args["learning_rate"])
    scheduler = StepLR(optimizer, step_size=1, gamma=args["scheduler_gamma"])
    if args["load_optimizer_scheduler_checkpoint_path"]:
        print(f"Loading {args['load_optimizer_scheduler_checkpoint_path']}")
        checkpoint_state_dicts = torch.load(
            args["load_optimizer_scheduler_checkpoint_path"], 
            weights_only=True
        )
        optimizer_state_dict = checkpoint_state_dicts["optimizer_state_dict"]
        scheduler_state_dict = checkpoint_state_dicts["scheduler_state_dict"]
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)

    # core loop of epochs
    for epoch_idx in range(1, args["num_epochs"] + 1):
        # shuffle data differently at each epoch across all processes
        train_sampler.set_epoch(epoch=epoch_idx)

        train_epoch(
            rank=rank,
            current_epoch=epoch_idx,
            model=model,    
            dataloader=train_loader,
            optimizer=optimizer,
            args=args
        )
        scheduler.step()
        dist.barrier()  # use a barrier to make sure training is done on all ranks
        
        eval_epoch(
            rank=rank,
            model=model,
            current_epoch=epoch_idx,
            dataloader=eval_loader,
            args=args
        )
        dist.barrier()

        if (
            epoch_idx == 1 
            or epoch_idx == args["num_epochs"] 
            or epoch_idx % args["save_every_epochs"] == 0
        ):
            if rank == 0:
                adapter_checkpoint_dir = os.path.join(
                    args["save_checkpoint_dir"], 
                    f"adapter_checkpoint_{epoch_idx}"
                )
                model.module.save_pretrained(adapter_checkpoint_dir)
                print(f"Saving {adapter_checkpoint_dir}")

                optimizer_scheduler_checkpoint_path = os.path.join(
                    args["save_checkpoint_dir"], 
                    f"optimizer_scheduler_checkpoint_{epoch_idx}.pt"
                )
                torch.save(
                    {
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }, 
                    optimizer_scheduler_checkpoint_path
                )
                print(f"Saving {optimizer_scheduler_checkpoint_path}")

            dist.barrier()

    cleanup()


def train_distributed(
        args: Dict[str, Any]  # replace **kwargs for compatibility with spawn
):
    """
    Core training process across multiple devices with epochs of training and 
    inter-epoch evaluation.
    """
    torch.multiprocessing.spawn(
        train_on_device, 
        args=(args["world_size"], args),
        nprocs=args["world_size"],
        join=True
    )


if __name__ == '__main__':
    # suppress messages from AutoTokenizer parallelism and Graphein respectively
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["LOGURU_LEVEL"] = "INFO"

    parsed_args = argParser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # restrict GPU visibility
    parsed_args.world_size = torch.cuda.device_count()  # use up all visible GPUs

    torch.manual_seed(parsed_args.random_seed)
    torch.cuda.manual_seed(parsed_args.random_seed)
    
    # initialize checkpoint directory
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    parsed_args.save_checkpoint_dir = os.path.join(
        parsed_args.save_checkpoint_dir, 
        f"checkpoints_{timestamp}"
    )
    if not os.path.exists(parsed_args.save_checkpoint_dir):
        os.mkdir(parsed_args.save_checkpoint_dir)
    
    print("####################")
    for key, value in parsed_args.__dict__.items(): 
        print(f"{key}: {value}")
    print("####################")

    train_distributed(parsed_args.__dict__)
