import argparse
from pathlib import Path
import numpy as np
import os
import shutil

def list_subfolders(target_dir):
    path = Path(target_dir)
    
    # Check if the path exists and is a directory
    if not path.is_dir():
        print(f"Error: '{target_dir}' is not a valid directory.")
        return

    print(f"Iterating direct subfolders in: {path.absolute()}\n")
    
    for entry in path.iterdir():
        if entry.is_dir():
            source_file = os.path.join(str(entry), 'meshes', 'model.obj')
            if not os.path.isfile(source_file):
                print(f"Warning: '{source_file}' does not exist. Skipping.")
                continue

            dest_file = os.path.join(output_folder, f"{entry.name}.obj")

            shutil.copyfile(source_file, dest_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate direct subfolders.")
    # parser.add_argument("--input_folder", default='', help="Path to the input folder")
    # parser.add_argument("--output_folder", default='', help="Path to the input folder")
    
    args = parser.parse_args()

    input_folder = '/PHShome/yl535/project/python/datasets/google_scanned_objects'
    output_folder = '/PHShome/yl535/project/python/datasets/google_scanned_objects_preprocessed'

    os.makedirs(output_folder, exist_ok=True)

    list_subfolders(input_folder)