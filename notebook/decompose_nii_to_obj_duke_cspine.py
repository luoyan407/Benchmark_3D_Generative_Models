import numpy as np
import os
from nilearn import image
import nibabel as nib
import argparse
import pathlib

def find_nifti_pairs(input_folder_path):
    """
    Finds pairs of (image.nii.gz, image_SEG.nii.gz) in a directory.
    """
    input_path = pathlib.Path(input_folder_path)
    
    # 1. Find all segmentation files first (files ending in _SEG.nii.gz)
    # This acts as our "anchor" to find the pairs
    seg_files = input_path.glob('*_SEG.nii.gz')
    
    pairs = []

    print(f"Searching in: {input_path.resolve()}\n")

    for seg_path in seg_files:
        # 2. Construct the expected base image filename
        # We replace '_SEG.nii.gz' with '.nii.gz' to get the base name
        image_filename = seg_path.name.replace('_SEG.nii.gz', '.nii.gz')
        image_path = input_path / image_filename

        # 3. Verify that the corresponding base image actually exists
        if image_path.exists():
            pairs.append((image_path, seg_path))
        else:
            print(f"⚠️ Warning: Found mask {seg_path.name}, but missing base image {image_filename}")

    return pairs

def segment_and_crop_objects(intensity_nii_path, mask_nii_path, output_dir="segmented_objects"):
    """
    Splits a multi-label NIfTI mask into individual segmented NIfTI files.
    Applies the mask to the intensity image and crops to the object's bounding box.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data...")
    img_vol = image.load_img(intensity_nii_path)
    mask_vol = image.load_img(mask_nii_path)

    # This fixes the "affine mismatch" / "FOV errors"
    print("  Ensuring mask alignment...")
    mask_vol = image.resample_to_img(
        source_img=mask_vol,
        target_img=img_vol,
        interpolation="nearest"  # CRITICAL: Keeps labels as integers (0, 1, 2...), avoids blurring
    )

    filename = os.path.basename(intensity_nii_path)
    base_name = filename.split('.nii')[0] 
    
    # Get data to find unique labels
    mask_data = mask_vol.get_fdata()
    unique_labels = np.unique(mask_data)
    # Filter out background (0)
    object_ids = unique_labels[unique_labels != 0].astype(int)
    
    print(f"Found {len(object_ids)} objects: {object_ids}")
    
    for label_id in object_ids:
        print(f"Processing Object {label_id}...")
        
        # 1. Create a binary mask for the current object
        # binary_mask will now inherit the CORRECT affine from the resampled mask_vol
        binary_mask = image.math_img(
            f"img == {label_id}", 
            img=mask_vol
        )
        
        # 2. Apply mask to original intensity image
        # This will now succeed because affines match
        masked_object = image.math_img(
            "img * mask", 
            img=img_vol, 
            mask=binary_mask
        )
        
        # 3. Crop to Bounding Box
        cropped_mask = image.crop_img(binary_mask)
        
        # Resample masked object to cropped geometry
        cropped_object = image.resample_to_img(masked_object, cropped_mask)
        
        # 4. Save to file
        save_path_img = os.path.join(output_dir, f"{base_name}_{label_id}.nii.gz")
        save_path_mask = os.path.join(output_dir, f"{base_name}_{label_id}_mask.nii.gz")
        
        nib.save(cropped_object, save_path_img)
        nib.save(cropped_mask, save_path_mask)
        
        print(f"  Saved Image: {save_path_img} (Shape: {cropped_object.shape})")

def get_image_mask_pairs(root_dir, type="airways"):
    """
    Traverses subfolders named [id] and finds matching CT/Mask pairs.
    """
    data_pairs = []
    
    # 1. Get all subfolders in the root directory
    # We filter to ensure we only look at directories
    subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # 2. Sort them numerically (so we process 1, 2, ... 10, instead of 1, 10, 2)
    # If folder names are not pure integers, this falls back to string sorting
    try:
        subfolders.sort(key=lambda x: int(x))
    except ValueError:
        subfolders.sort()

    print(f"Found {len(subfolders)} patient folders.")

    for case_id in subfolders:
        case_path = os.path.join(root_dir, case_id)
        
        # 3. Construct specific filenames based on the pattern
        # Pattern: [id]_CT_HR.nii.gz AND [id]_CT_HR_label_airways.nii.gz
        img_name = f"{case_id}_CT_HR.nii.gz"
        mask_name = f"{case_id}_CT_HR_label_{type}.nii.gz"
        
        img_full_path = os.path.join(case_path, img_name)
        mask_full_path = os.path.join(case_path, mask_name)
        
        # 4. Verify files exist before adding to list
        if os.path.exists(img_full_path) and os.path.exists(mask_full_path):
            data_pairs.append((img_full_path, mask_full_path))
        else:
            print(f"Warning: Missing files in folder {case_id}")
            # Optional: Check which one is missing for debugging
            if not os.path.exists(img_full_path): print(f"  - Missing: {img_name}")
            if not os.path.exists(mask_full_path): print(f"  - Missing: {mask_name}")

    return data_pairs

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="lungs", type=str,
                    help='Type of dataset/task ("airways" or "lungs")')
args = parser.parse_args()

type = args.type  # "airways" or "lungs"
input_folder = "/PHShome/yl535/project/python/datasets/duke_cspine/" 
output_dir = f"/PHShome/yl535/project/python/datasets/duke_cspine/segmented" 

# pairs = get_image_mask_pairs(input_folder, type=type)

pairs = find_nifti_pairs(input_folder)

# Example loop to process them
for img_path, mask_path in pairs:
    img_path = str(img_path)
    mask_path = str(mask_path)

    filename = os.path.basename(img_path)
    base_name = filename.split('.nii')[0] 

    print(f"Processing: {os.path.basename(img_path)}, {os.path.basename(mask_path)}")

    segment_and_crop_objects(img_path, mask_path, output_dir=output_dir)