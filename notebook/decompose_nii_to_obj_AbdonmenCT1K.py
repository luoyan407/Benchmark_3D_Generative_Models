import numpy as np
import os
from nilearn import image
import nibabel as nib
import argparse

def segment_and_crop_objects(intensity_nii_path, mask_nii_path, output_dir="segmented_objects"):
    """
    Splits a multi-label NIfTI mask into individual segmented NIfTI files.
    Applies the mask to the intensity image and crops to the object's bounding box.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {os.path.basename(intensity_nii_path)}...")
    img_vol = image.load_img(intensity_nii_path)
    
    # --- FIX STARTS HERE ---
    # Load raw mask
    raw_mask_vol = image.load_img(mask_nii_path)
    
    # Resample mask to match the intensity image's shape and affine.
    # We use 'nearest' interpolation to preserve integer labels (0, 1, 2...).
    print("  Resampling mask to match image geometry...")
    mask_vol = image.resample_to_img(raw_mask_vol, img_vol, interpolation='nearest')
    # --- FIX ENDS HERE ---

    # Extract base name (handling .nii or .nii.gz)
    filename = os.path.basename(intensity_nii_path)
    base_name = filename.split('.nii.gz')[0] 
    
    # Get data to find unique labels
    mask_data = mask_vol.get_fdata()
    unique_labels = np.unique(mask_data)
    
    # Filter out background (0)
    object_ids = unique_labels[unique_labels != 0].astype(int)
    
    print(f"Found {len(object_ids)} objects: {object_ids}")
    
    for label_id in object_ids:
        # 1. Create a binary mask for the current object
        # Note: math_img works now because img_vol and mask_vol have same shape
        binary_mask = image.math_img(
            f"img == {label_id}", 
            img=mask_vol
        )
        
        # 2. Apply mask to original intensity image
        masked_object = image.math_img(
            "img * mask", 
            img=img_vol, 
            mask=binary_mask
        )
        
        # 3. Crop to Bounding Box
        cropped_mask = image.crop_img(binary_mask)
        
        # Resample the masked object to the cropped_mask geometry
        cropped_object = image.resample_to_img(masked_object, cropped_mask)
        
        # 4. Save to file
        save_path_img = os.path.join(output_dir, f"{base_name}_{label_id}.nii.gz")
        save_path_mask = os.path.join(output_dir, f"{base_name}_{label_id}_mask.nii.gz")
        
        nib.save(cropped_object, save_path_img)
        nib.save(cropped_mask, save_path_mask)
        
        print(f"  Saved: {os.path.basename(save_path_img)} (Shape: {cropped_object.shape})")

def get_image_mask_pairs(img_dir, label_dir):
    data_pairs = []
    
    # Get all files in images directory that end with .nii or .nii.gz
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    # Sort to ensure processing order
    img_files.sort()

    print(f"Scanning {len(img_files)} files in image directory...")

    for img_filename in img_files:
        # --- FIX: Skip hidden macOS metadata files ---
        if img_filename.startswith("._"):
            continue
        # ---------------------------------------------

        # This splits 'img0001.nii' -> 'img0001'
        base_name = img_filename.split('.nii.gz')[0] 

        # Construct the full path for the image
        img_full_path = os.path.join(img_dir, img_filename)
        
        label_full_path = os.path.join(label_dir, base_name, f'{base_name}_lab.nii.gz')
        
        # Verify files exist before adding to list
        if os.path.exists(label_full_path):
            data_pairs.append((img_full_path, label_full_path))
        else:
            print(f"Warning: Missing label file for {img_filename}")

    return data_pairs

# --- MAIN EXECUTION ---

# Define the two specific folders
images_dir = "/PHShome/yl535/project/python/datasets/AbdomenCT-1K-Image/images/"
labels_dir = "/PHShome/yl535/project/python/datasets/AbdomenCT-1K-Image/seg_mask/"

# Define output directory
output_dir = "/PHShome/yl535/project/python/datasets/AbdomenCT-1K-Image/segmented" 

# Check if input directories exist to avoid crash
if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
    print("Error: Input directories not found.")
    print(f"Checked: {images_dir}")
    print(f"Checked: {labels_dir}")
else:
    # Get pairs
    pairs = get_image_mask_pairs(images_dir, labels_dir)

    print(f"\nSuccessfully loaded {len(pairs)} pairs.")

    # Process pairs
    for img_path, mask_path in pairs:
        print(f"Processing Pair: {os.path.basename(img_path)} | {os.path.basename(mask_path)}")
        # segment_and_crop_objects(img_path, mask_path, output_dir=output_dir)

        try:
            # Try to process the file
            segment_and_crop_objects(img_path, mask_path, output_dir=output_dir)
            
        except Exception as e:
            # If ANY error occurs (empty file, bad header, shape mismatch), print and skip
            print(f"!!! SKIPPING {os.path.basename(img_path)} !!!")
            print(f"    Error details: {e}\n")
            continue