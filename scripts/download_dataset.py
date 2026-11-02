# -------------------------------------------------
# scripts/download_dataset.py
# -------------------------------------------------
import argparse
import urllib.request
from pathlib import Path

# Base URL for the official Hugging Face dataset (raw files)
HF_BASE = "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/main/"

def download(split: str, out_dir: Path) -> Path:
    filename = f"freshretailnet_{split}.parquet"
    url = f"{HF_BASE}{filename}"
    out_path = out_dir / filename
    print(f"Downloading {url} → {out_path}")
    urllib.request.urlretrieve(url, out_path)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "test", "val"])
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        download(args.split, out_dir)
    except Exception as e:
        print("❌ Download failed:", e)
        print(
            "If the URL has changed, open the HuggingFace dataset page\n"
            "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K\n"
            "and copy the raw file URL for the requested split."
        )
        raise

if __name__ == "__main__":
    main()
