import numpy as np
import os
from nilearn import image
import nibabel as nib

import torch
import matplotlib.pyplot as plt
from PIL import Image
import requests
from transformers import pipeline
from pathlib import Path
from typing import List, Tuple
import argparse

import sys
import imageio
import uuid
from direct3d.pipeline import Direct3dPipeline



# # Set up paths
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# NOTEBOOK_PATH = os.path.join(str(PROJECT_ROOT), "notebook")

# # Insert at index 0 so it searches 'notebook' BEFORE the local folder
# if NOTEBOOK_PATH not in sys.path:
#     sys.path.insert(0, NOTEBOOK_PATH)

# # Explicitly check if the file exists where we expect it
# expected_file = os.path.join(NOTEBOOK_PATH, "inference.py")
# if not os.path.exists(expected_file):
#     print(f"CRITICAL ERROR: {expected_file} not found!")

# from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_mask, load_single_mask, display_image, make_scene, interactive_visualizer

def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image


def load_mask(path):
    mask = load_image(path)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    return mask

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

def show_masks(image, masks):
    """
    Helper function to display the image with masks overlaid.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Iterate through each mask and overlay it
    for mask in masks:
        # Convert PIL mask to numpy array (boolean)
        m = np.array(mask) > 0 
        
        # Generate a random color for this mask
        color = np.concatenate([np.random.random(3), [0.6]]) # [R, G, B, Alpha]
        
        # Create a colored mask image
        h, w = m.shape
        mask_image = m.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        # Overlay the mask
        ax.imshow(mask_image)

    plt.axis('off')
    plt.show()

def get_object_middle_index(mask_data, label_id, axis=2):
    """
    Function 1: Determines the middle slice index for a SPECIFIC object label.
    It finds the bounding box of the label and calculates the center.
    """
    # Find the indices where the mask equals the current label
    # np.where returns a tuple of arrays (z, y, x)
    object_indices = np.where(mask_data == label_id)
    
    # Check if object exists
    if len(object_indices[0]) == 0:
        return None
        
    # Get indices for the specific axis
    axis_indices = object_indices[axis]
    
    # Calculate center: (min + max) // 2
    min_idx = np.min(axis_indices)
    max_idx = np.max(axis_indices)
    
    middle_index = (min_idx + max_idx) // 2
    
    return middle_index

def to_uint8_image(arr):
    # 1. Normalize array to range [0, 1]
    # (Avoid division by zero if max == min)
    if arr.max() == arr.min():
        arr_normalized = np.zeros_like(arr)
    else:
        arr_normalized = (arr - arr.min()) / (arr.max() - arr.min())
    
    # 2. Scale to [0, 255]
    arr_scaled = arr_normalized * 255
    
    # 3. Cast to uint8
    return arr_scaled.astype(np.uint8)

def extract_all_objects_middle_slices(intensity_nii_path, mask_nii_path, axis=2):
    # Load data
    img_nii = nib.load(intensity_nii_path)
    mask_nii = nib.load(mask_nii_path)
    
    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    
    # Handle 4D data
    if img_data.ndim == 4:
        img_data = img_data[..., 0]
    if mask_data.ndim == 4:
        mask_data = mask_data[..., 0]

    unique_labels = np.unique(mask_data)
    object_ids = unique_labels[unique_labels != 0].astype(int)
    
    print(f"Found {len(object_ids)} objects. Extracting middle slices...")
    
    results = []
    for label_id in object_ids:
        mid_idx = get_object_middle_index(mask_data, label_id, axis)
        
        if mid_idx is None:
            continue
            
        if axis == 0:
            slice_img = img_data[mid_idx, :, :]
            slice_mask = mask_data[mid_idx, :, :]
        elif axis == 1:
            slice_img = img_data[:, mid_idx, :]
            slice_mask = mask_data[:, mid_idx, :]
        else: # axis == 2
            slice_img = img_data[:, :, mid_idx]
            slice_mask = mask_data[:, :, mid_idx]
        
        # --- FIX STARTS HERE ---
        # Create a binary mask for the SPECIFIC label_id
        # The raw slice_mask contains values like 1, 2, 3... we need 0/1 for the current object
        slice_mask = (slice_mask == label_id).astype(np.uint8)
        # --- FIX ENDS HERE ---

        # 1. Expand Image to 3 Channels (RGB)
        slice_img = np.repeat(slice_img[:, :, np.newaxis], 3, axis=2)
        slice_img = to_uint8_image(slice_img)
        
        slice_mask = np.squeeze(slice_mask)
        
        results.append((slice_img, slice_mask, label_id))
            
    return results

def get_all_directories_pathlib(input_folder):
    path = Path(input_folder)
    # .rglob('*') recursively finds everything; we filter for directories
    directories = [str(f) for f in path.rglob('*') if f.is_dir()]
    return directories

def get_all_images_pathlib(input_folder):
    extensions = ['.jpeg', '.jpg', '.png', '.bmp', '.tiff']
    path = Path(input_folder)
    file_list = []
    for file_path in path.rglob('*'):
        # file_list.append(str(file_path))
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            file_list.append(str(file_path))
    return file_list

def visualize_segmented_object(segmented_slice, label_id):
    """
    Function 3: Visualizes the segmented object.
    """
    # Rotate for standard orientation
    to_show = np.rot90(segmented_slice)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(to_show, cmap='gray')
    plt.title(f"Object Label: {label_id}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def find_image_mask_pairs(input_folder: str, recursive: bool = True) -> List[Tuple[Path, Path]]:
    p = Path(input_folder)
    it = p.rglob("*.nii.gz") if recursive else p.glob("*.nii.gz")

    images = {}  # key -> image path
    masks  = {}  # key -> mask path

    for f in it:
        name = f.name
        if name.endswith("_mask.nii.gz"):
            key = name[:-len("_mask.nii.gz")]          # e.g., "10_CT_HR_1"
            masks[key] = f
        else:
            key = name[:-len(".nii.gz")]               # e.g., "10_CT_HR_1"
            images[key] = f

    pairs = [(images[k], masks[k]) for k in sorted(images.keys() & masks.keys())]
    return pairs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference")
    
    parser.add_argument(
        '--input_folder', 
        type=str, 
        required=True,
        help='Path to the folder containing input NIfTI files.'
    )
    parser.add_argument(
        '--output_folder', 
        type=str, 
        required=True,
        help='Path to the folder to save output results.'
    )
    
    args = parser.parse_args()

    # config_path = args.config_path # f"/PHShome/yl535/project/python/sam_3d/sam-3d-objects/checkpoints/checkpoints/pipeline.yaml"
    # type = "lungs"  # "airways" or "lungs"
    input_folder = args.input_folder # f"/PHShome/yl535/project/python/datasets/AeroPath/lungs_segmented" 
    output_dir = args.output_folder  # f"/PHShome/yl535/project/python/sam_3d/sam-3d-objects/results_AeroPath/lungs" 

    # inference = Inference(config_path, compile=False)

    # device = 0 if torch.cuda.is_available() else -1
    # generator = pipeline(
    #     "mask-generation", 
    #     model="facebook/sam2-hiera-large", 
    #     device=device,
    #     torch_dtype=torch.float32 
    # )

    pipeline = Direct3dPipeline.from_pretrained("DreamTechAI/Direct3D")
    pipeline.to("cuda")

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    all_samples = get_all_images_pathlib(os.path.join(input_folder, 'images', 'test'))

    for img_path in all_samples:
        class_name = os.path.basename(os.path.dirname(img_path))
        file_name = os.path.basename(img_path).split(".")[0]
        output_file = f"{output_dir}/{file_name}.ply"
        
        mask_path = os.path.join(input_folder, 'masks', 'test', class_name, f'{file_name}.png')

        if os.path.exists(output_file):
            print(f"Skipping {file_name}: output already exists.")
            continue

        if (not os.path.exists(img_path)):
            print(f"Skipping {img_path}: image not found.")
            continue
        if (not os.path.exists(mask_path)):
            print(f"Skipping {mask_path}: mask not found.")
            continue
        image = Image.open(img_path)
        image_np = np.array(image)

        mask_obj = load_mask(mask_path)

        mesh = pipeline(
            img_path,
            remove_background=False, 
            mc_threshold=-1.0,
            guidance_scale=4.0,
            num_inference_steps=50,
        )["meshes"][0]

        mesh.export(f"{output_file}")

        # try:
        #     output = inference(image_np, mask_obj, seed=42)
        #     output["gs"].save_ply(output_file)

        #     scene_gs = make_scene(output)
        #     scene_gs = ready_gaussian_for_video_rendering(scene_gs)

        #     video = render_video(
        #         scene_gs,
        #         r=1,
        #         fov=60,
        #         pitch_deg=15,
        #         yaw_start_deg=-45,
        #         resolution=512,
        #     )["color"]

        #     # save video as gif
        #     imageio.mimsave(
        #         os.path.join(f"{output_dir}/{file_name}.gif"),
        #         video,
        #         format="GIF",
        #         duration=1000 / 30,  # default assuming 30fps from the input MP4
        #         loop=0,  # 0 means loop indefinitely
        #     )
            
        # except ValueError as e:
        #     if "max() arg is an empty sequence" in str(e):
        #         print(f"Skipping {file_name}: Background removal resulted in empty image.")
        #     else:
        #         print(f"Error processing {file_name}: {e}")
        # except Exception as e:
        #     print(f"Unexpected error on {file_name}: {e}")