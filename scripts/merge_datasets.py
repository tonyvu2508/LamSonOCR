"""Merge all datasets into one combined dataset."""
import csv
from pathlib import Path

def merge_datasets():
    data_dir = Path("/Volumes/SpaceX/WorkSpace/python/LamSonOcr/data")
    output_dir = data_dir / "all_train"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to merge
    csv_paths = [
        data_dir / "train" / "labels.csv",
        data_dir / "val" / "labels.csv",
        data_dir / "etl_train" / "labels.csv",
    ]
    
    merged_rows = []
    
    for path in csv_paths:
        if path.exists():
            print(f"Reading {path}...")
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    merged_rows.append(row)
                    count += 1
                print(f"  Loaded {count} rows")
        else:
            print(f"Warning: {path} does not exist. Skipping.")
            
    output_csv = output_dir / "labels.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(merged_rows)
        
    print(f"✅ Merged all datasets. Total samples: {len(merged_rows)} -> Saved to {output_csv}")

if __name__ == "__main__":
    merge_datasets()
