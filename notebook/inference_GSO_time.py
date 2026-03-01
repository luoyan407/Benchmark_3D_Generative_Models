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
import time  # <--- Added
import csv   # <--- Added
from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(os.path.join(str(PROJECT_ROOT), "checkpoints"))

def get_image_mask_pairs(root_dir, type="airways"):
    """
    Traverses subfolders named [id] and finds matching CT/Mask pairs.
    """
    data_pairs = []
    
    # 1. Get all subfolders in the root directory
    subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # 2. Sort them numerically 
    try:
        subfolders.sort(key=lambda x: int(x))
    except ValueError:
        subfolders.sort()

    print(f"Found {len(subfolders)} patient folders.")

    for case_id in subfolders:
        case_path = os.path.join(root_dir, case_id)
        img_name = f"{case_id}_CT_HR.nii.gz"
        mask_name = f"{case_id}_CT_HR_label_{type}.nii.gz"
        
        img_full_path = os.path.join(case_path, img_name)
        mask_full_path = os.path.join(case_path, mask_name)
        
        if os.path.exists(img_full_path) and os.path.exists(mask_full_path):
            data_pairs.append((img_full_path, mask_full_path))
        else:
            print(f"Warning: Missing files in folder {case_id}")
            if not os.path.exists(img_full_path): print(f"  - Missing: {img_name}")
            if not os.path.exists(mask_full_path): print(f"  - Missing: {mask_name}")

    return data_pairs

def show_masks(image, masks):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for mask in masks:
        m = np.array(mask) > 0 
        color = np.concatenate([np.random.random(3), [0.6]]) # [R, G, B, Alpha]
        h, w = m.shape
        mask_image = m.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    plt.axis('off')
    plt.show()

def get_object_middle_index(mask_data, label_id, axis=2):
    object_indices = np.where(mask_data == label_id)
    if len(object_indices[0]) == 0:
        return None
    axis_indices = object_indices[axis]
    min_idx = np.min(axis_indices)
    max_idx = np.max(axis_indices)
    middle_index = (min_idx + max_idx) // 2
    return middle_index

def to_uint8_image(arr):
    if arr.max() == arr.min():
        arr_normalized = np.zeros_like(arr)
    else:
        arr_normalized = (arr - arr.min()) / (arr.max() - arr.min())
    arr_scaled = arr_normalized * 255
    return arr_scaled.astype(np.uint8)

def extract_all_objects_middle_slices(intensity_nii_path, mask_nii_path, axis=2):
    img_nii = nib.load(intensity_nii_path)
    mask_nii = nib.load(mask_nii_path)
    
    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    
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
        
        slice_mask = (slice_mask == label_id).astype(np.uint8)

        slice_img = np.repeat(slice_img[:, :, np.newaxis], 3, axis=2)
        slice_img = to_uint8_image(slice_img)
        slice_mask = np.squeeze(slice_mask)
        results.append((slice_img, slice_mask, label_id))
            
    return results

def get_all_directories_pathlib(input_folder):
    path = Path(input_folder)
    directories = [str(f) for f in path.rglob('*') if f.is_dir()]
    return directories

def visualize_segmented_object(segmented_slice, label_id):
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

    images = {}  
    masks  = {} 

    for f in it:
        name = f.name
        if name.endswith("_mask.nii.gz"):
            key = name[:-len("_mask.nii.gz")]          
            masks[key] = f
        else:
            key = name[:-len(".nii.gz")]               
            images[key] = f

    pairs = [(images[k], masks[k]) for k in sorted(images.keys() & masks.keys())]
    return pairs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference")
    
    parser.add_argument(
        '--config_path', 
        type=str, 
        required=True,
        help='Path to the configuration YAML file.'
    )
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

    config_path = args.config_path 
    input_folder = args.input_folder 
    output_dir = args.output_folder 

    inference = Inference(config_path, compile=False)

    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "mask-generation", 
        model="facebook/sam2-hiera-large", 
        device=device,
        torch_dtype=torch.float32 
    )

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    all_samples = get_all_directories_pathlib(input_folder)

    # --- STATISTICS SETUP ---
    inference_times = []
    # ------------------------

    for img_path in all_samples:
        file_path = os.path.join(img_path, 'thumbnails', "0.jpg")

        if not os.path.exists(file_path):
            print(f"Skipping {img_path}: Thumbnail not found.")
            continue
        image = Image.open(file_path)
        IMAGE_NAME = os.path.basename(img_path) 

        outputs = generator(image, points_per_batch=64)

        masks = [x.detach().numpy() for x in outputs["masks"]] 
        image_np = np.array(image)

        mask_index = 1

        try:
            # --- START TIMING ---
            start_time = time.time()
            
            output = inference(image_np, masks[mask_index], seed=42)
            
            end_time = time.time()
            duration = end_time - start_time
            inference_times.append(duration)
            print(f"Sample {IMAGE_NAME} inference time: {duration:.4f}s")
            # --- END TIMING ---

            output["gs"].save_ply(f"{output_dir}/{IMAGE_NAME}.ply")

            scene_gs = make_scene(output)
            scene_gs = ready_gaussian_for_video_rendering(scene_gs)

            video = render_video(
                scene_gs,
                r=1,
                fov=60,
                pitch_deg=15,
                yaw_start_deg=-45,
                resolution=512,
            )["color"]

            # save video as gif
            imageio.mimsave(
                os.path.join(f"{output_dir}/{IMAGE_NAME}.gif"),
                video,
                format="GIF",
                duration=1000 / 30,  
                loop=0,  
            )
            
        except ValueError as e:
            if "max() arg is an empty sequence" in str(e):
                print(f"Skipping {IMAGE_NAME}: Background removal resulted in empty image.")
            else:
                print(f"Error processing {IMAGE_NAME}: {e}")
        except Exception as e:
            print(f"Unexpected error on {IMAGE_NAME}: {e}")

    # --- SAVE STATISTICS TO CSV ---
    if inference_times:
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        
        stats_file = os.path.join(output_dir, "inference_stats_GSO.csv") # Changed filename slightly to distinguish
        
        print(f"\n--- Statistics ---")
        print(f"Total successful samples: {len(inference_times)}")
        print(f"Mean inference time: {mean_time:.4f} s")
        print(f"Std dev inference time: {std_time:.4f} s")
        
        with open(stats_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value (seconds)"])
            writer.writerow(["Mean", mean_time])
            writer.writerow(["Standard Deviation", std_time])
            writer.writerow(["Total Samples", len(inference_times)])
            
        print(f"Statistics saved to {stats_file}")
    else:
        print("\nNo successful inference runs recorded. No statistics generated.")