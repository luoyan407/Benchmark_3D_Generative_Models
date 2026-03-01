import argparse
from pathlib import Path
import numpy as np
import os
import shutil

def list_subfolders(target_dir):
    path = Path(target_dir)

    for entry in path.rglob('*.obj'):
        source_file = str(entry)
        dest_file = os.path.join(output_folder, f"{entry.name}")
        shutil.copyfile(source_file, dest_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate direct subfolders.")
    # parser.add_argument("--input_folder", default='', help="Path to the input folder")
    # parser.add_argument("--output_folder", default='', help="Path to the input folder")
    
    args = parser.parse_args()

    input_folder = '/PHShome/yl535/project/python/datasets/animal3d/obj_files/test'
    output_folder = '/PHShome/yl535/project/python/datasets/animal3d/preprocessed'

    os.makedirs(output_folder, exist_ok=True)

    list_subfolders(input_folder)