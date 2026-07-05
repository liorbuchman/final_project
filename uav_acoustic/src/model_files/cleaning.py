import os
import pandas as pd
from pathlib import Path

def clean_and_standardize_raw_data(raw_dir):
    raw_path = Path(raw_dir)
    mapping = []

    for label in ['drone', 'not_drone']:
        label_folder = raw_path / label
        if not label_folder.exists():
            continue

        # sort files to ensure consistent renaming
        files = sorted
        (list(label_folder.glob("*.wav")))
        
        print(f"[INFO] Renaming {len(files)} files in '{label}' folder...")
        
        for i, old_path in enumerate(files):
            new_name = f"{label}_{i:04d}.wav"
            new_path = label_folder / new_name
            
            # mapping for record keeping
            mapping.append({
                "original_name": old_path.name,
                "new_name": new_name,
                "label": label
            })
            
            # change the name in place
            old_path.rename(new_path)

    # save mapping file in the labels directory
    mapping_df = pd.DataFrame(mapping)
    output_dir = raw_path.parent / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(output_dir / "original_files_mapping.csv", index=False)
    
    print(f"[SUCCESS] All files renamed. Mapping saved to: {output_dir / 'original_files_mapping.csv'}")

if __name__ == "__main__":
    RAW_DIR = r"C:\final_project\uav_acoustic\data\raw"
    clean_and_standardize_raw_data(RAW_DIR)