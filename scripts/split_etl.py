"""Split large ETL zip files into 99MB chunks to bypass GitHub file size limits."""
import os
from pathlib import Path

CHUNK_SIZE = 99 * 1024 * 1024  # 99 MB

def split_file(file_path: Path, output_dir: Path):
    if not file_path.exists():
        return
        
    file_size = file_path.stat().st_size
    print(f"Splitting {file_path.name} ({file_size / (1024*1024):.2f} MB)...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_num = 0
    with open(file_path, 'rb') as infile:
        while True:
            chunk = infile.read(CHUNK_SIZE)
            if not chunk:
                break
                
            chunk_name = output_dir / f"{file_path.name}.part_{chunk_num:03d}"
            with open(chunk_name, 'wb') as outfile:
                outfile.write(chunk)
            print(f"  Created chunk: {chunk_name.name} ({len(chunk) / (1024*1024):.2f} MB)")
            chunk_num += 1

def main():
    project_root = Path(__file__).resolve().parent.parent
    etl_dir = project_root / "ETL"
    parts_dir = project_root / "ETL_parts"
    
    if not etl_dir.exists():
        print("ETL directory not found.")
        return
        
    zip_files = list(etl_dir.glob("*.zip"))
    if not zip_files:
        print("No .zip files found in ETL/ directory.")
        return
        
    print(f"Found {len(zip_files)} zip files to split.")
    for zip_file in zip_files:
        split_file(zip_file, parts_dir)
        
    print("\n✅ Splitting complete. Parts saved to ETL_parts/")

if __name__ == "__main__":
    main()
