from datasets import load_dataset
import os

# Pull down all splits
ds = load_dataset("habdine/Prot2Text-Data")

os.makedirs("data", exist_ok=True)
for split in ("train", "validation", "test"):
    # Hugging Face names the files exactly train-00000-of-00001.parquet, etc.
    df = ds[split].to_pandas()
    df.to_csv(f"data/{split}.csv", index=False)
    print(f"Wrote data/{split}.csv ({len(df)} rows)")
