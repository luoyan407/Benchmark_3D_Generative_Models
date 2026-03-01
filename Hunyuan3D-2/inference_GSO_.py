import os
import argparse
import torch
import numpy as np
import trimesh
from PIL import Image
from hi3dgen.pipelines import Hi3DGenPipeline
from huggingface_hub import snapshot_download

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

from glob import glob
from typing import Any, Union

# --- Configuration ---
os.environ['SPCONV_ALGO'] = 'native'
MAX_SEED = np.iinfo(np.int32).max
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')

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
        
        # 1. Expand Image to 3 Channels (RGB)
        slice_img = np.repeat(slice_img[:, :, np.newaxis], 3, axis=2)
        slice_img = to_uint8_image(slice_img)
        slice_mask = np.squeeze(slice_mask)
        
        results.append((slice_img, slice_mask, label_id))
            
    return results

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

def cache_weights(weights_dir: str):
    """
    Downloads and caches the required model weights from Hugging Face.
    """
    os.makedirs(weights_dir, exist_ok=True)
    model_ids = [
        "Stable-X/trellis-normal-v0-1",
        "Stable-X/yoso-normal-v1-8-1",
        "ZhengPeng7/BiRefNet",
    ]
    
    for model_id in model_ids:
        local_path = os.path.join(weights_dir, model_id.split("/")[-1])
        if os.path.exists(local_path):
            print(f"Weights already cached: {model_id}")
            continue
            
        print(f"Downloading weights: {model_id}...")
        snapshot_download(
            repo_id=model_id, 
            local_dir=local_path, 
            force_download=False
        )
        print(f"Cached at: {local_path}")

def load_models(device='cuda'):
    print("Loading Hi3DGen Pipeline...")
    pipeline_path = os.path.join(WEIGHTS_DIR, "trellis-normal-v0-1")
    
    if not os.path.exists(pipeline_path):
        pipeline_path = "Stable-X/trellis-normal-v0-1"
        
    hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained(pipeline_path)
    hi3dgen_pipeline.to(device)

    print("Loading NiRNE Normal Predictor...")
    
    nirne_predictor = torch.hub.load(
        "lzt02/NiRNE", "NiRNE",
        trust_repo=True,
        local_cache_dir=WEIGHTS_DIR
    )
    # Check if NiRNE is a local folder
    # if os.path.isdir("NiRNE"):
    #     nirne_predictor = torch.hub.load(
    #         "NiRNE", 
    #         "NiRNE", 
    #         source='local',
    #         trust_repo=True, 
    #         local_cache_dir=WEIGHTS_DIR,
    #         # variant=None,               # Forces it to not look for 'fp16' specific files
    #         torch_dtype=torch.float32   # Ensures it loads as standard 32-bit float
    #     )
    # else:
    #     print("Local 'NiRNE' folder not found. Attempting to load from GitHub...")
    #     nirne_predictor = torch.hub.load(
    #         "Stable-X/StableNormal",
    #         "NiRNE", 
    #         trust_repo=True, 
    #         local_cache_dir=WEIGHTS_DIR,
    #         # variant=None,
    #         torch_dtype=torch.float32
    #     )
    
    return hi3dgen_pipeline, nirne_predictor

def process_single_image(
    image_path, 
    output_path, 
    pipeline, 
    normal_predictor,
    seed=-1,
    ss_steps=50,
    ss_strength=3.0,
    slat_steps=6,
    slat_strength=3.0
):
    """
    Runs the reconstruction pipeline on a single image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # 1. Load Image
    print(f"Processing: {image_path}")
    image = Image.open(image_path).convert("RGB")

    # 2. Setup Seed
    if seed == -1:
        seed = np.random.randint(0, MAX_SEED)
    print(f"Using seed: {seed}")

    # 3. Preprocess Image (Resolution 1024 as per app.py)
    processed_image = pipeline.preprocess_image(image, resolution=1024)

    # 4. Normal Prediction (Resolution 768 as per app.py)
    print("Estimating Normals...")
    normal_image = normal_predictor(
        processed_image, 
        resolution=768, 
        match_input_resolution=True, 
        data_type='object'
    )

    # 5. Run Generation Pipeline
    print("Generating 3D Structure...")
    outputs = pipeline.run(
        normal_image,
        seed=seed,
        formats=["mesh"],
        preprocess_image=False, # Already preprocessed
        sparse_structure_sampler_params={
            "steps": ss_steps,
            "cfg_strength": ss_strength,
        },
        slat_sampler_params={
            "steps": slat_steps,
            "cfg_strength": slat_strength,
        },
    )

    # 6. Export Mesh
    generated_mesh = outputs['mesh'][0]
    trimesh_mesh = generated_mesh.to_trimesh(transform_pose=True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    trimesh_mesh.export(output_path)
    print(f"Successfully saved 3D mesh to: {output_path}")

def get_all_directories_pathlib(input_folder):
    path = Path(input_folder)
    # .rglob('*') recursively finds everything; we filter for directories
    directories = [str(f) for f in path.rglob('*') if f.is_dir()]
    return directories

def main():
    parser = argparse.ArgumentParser(description="Hi3DGen CLI: Single Image to 3D Mesh")
    
    # parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image")
    # parser.add_argument("--output", "-o", type=str, default="output.ply", help="Path to save output mesh (e.g., output.glb, output.obj)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    
    # Advanced Generation Settings
    parser.add_argument("--ss-steps", type=int, default=50, help="Sparse Structure Sampling Steps")
    parser.add_argument("--ss-strength", type=float, default=3.0, help="Sparse Structure Guidance Strength")
    parser.add_argument("--slat-steps", type=int, default=6, help="Structured Latent Sampling Steps")
    parser.add_argument("--slat-strength", type=float, default=3.0, help="Structured Latent Guidance Strength")
    
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
    parser.add_argument(
        '--slice_axis', 
        type=int, 
        default=2,
        help='Path to the folder to save output results.'
    )

    args = parser.parse_args()

    # 1. Ensure weights are present
    cache_weights(WEIGHTS_DIR)

    # 2. Load Models
    pipeline, normal_predictor = load_models()

    input_folder = args.input_folder
    output_dir = args.output_folder

    os.makedirs(output_dir, exist_ok=True)

    all_samples = get_all_directories_pathlib(input_folder)
    for img_path in all_samples:
        file_path = os.path.join(img_path, 'thumbnails', "0.jpg")

        if not os.path.exists(file_path):
            print(f"Skipping {img_path}: Thumbnail not found.")
            continue
        # image = Image.open(file_path)
        IMAGE_NAME = os.path.basename(img_path) # .split(".")[0]

        output_file = f"{output_dir}/{IMAGE_NAME}.ply"

        # extracted_scan_pil = Image.fromarray(file_path)

        # 1. Save the PIL image to a temporary path on disk
        # temp_img_path = os.path.join(output_dir, f"temp_{scan_path.name}.png")
        # extracted_scan_pil.save(temp_img_path)
        

        # 2. Run TripoSG with error handling
        try:
            # mesh = pipeline(image=temp_img_path)[0]
            # mesh.export(output_file)

            process_single_image(
                file_path,
                output_file,
                pipeline,
                normal_predictor,
                seed=args.seed,
                ss_steps=args.ss_steps,
                ss_strength=args.ss_strength,
                slat_steps=args.slat_steps,
                slat_strength=args.slat_strength
            )
            
            print(f"Saved mesh to {output_file}")
            
        except ValueError as e:
            if "max() arg is an empty sequence" in str(e):
                print(f"Skipping {file_path}: Background removal resulted in empty image.")
            else:
                print(f"Error processing {file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error on {file_path}: {e}")

    # pairs = find_image_mask_pairs(input_folder)
    # for scan_path,mask_path  in pairs:
    #     print("IMG :", scan_path.name)
    #     print("MASK:", mask_path.name)
    #     objs = extract_all_objects_middle_slices(scan_path, mask_path, axis=args.slice_axis)
    #     extracted_scan, extracted_mask = objs[0][0], objs[0][1]

    #     # Guard clause: Check if the extracted scan is empty/black
    #     if np.max(extracted_scan) == 0:
    #         print(f"Skipping {scan_path.name}: Extracted slice is empty.")
    #         continue

    #     output_file = f"{output_dir}/{scan_path.name}.ply"

    #     extracted_scan_pil = Image.fromarray(extracted_scan)

    #     # 1. Save the PIL image to a temporary path on disk
    #     temp_img_path = os.path.join(output_dir, f"temp_{scan_path.name}.png")
    #     extracted_scan_pil.save(temp_img_path)
        

    #     # 2. Run TripoSG with error handling
    #     try:
    #         # mesh = pipeline(image=temp_img_path)[0]
    #         # mesh.export(output_file)

    #         process_single_image(
    #             temp_img_path,
    #             output_file,
    #             pipeline,
    #             normal_predictor,
    #             seed=args.seed,
    #             ss_steps=args.ss_steps,
    #             ss_strength=args.ss_strength,
    #             slat_steps=args.slat_steps,
    #             slat_strength=args.slat_strength
    #         )
            
    #         print(f"Saved mesh to {output_file}")
            
    #     except ValueError as e:
    #         if "max() arg is an empty sequence" in str(e):
    #             print(f"Skipping {scan_path.name}: Background removal resulted in empty image.")
    #         else:
    #             print(f"Error processing {scan_path.name}: {e}")
    #     except Exception as e:
    #         print(f"Unexpected error on {scan_path.name}: {e}")

if __name__ == "__main__":
    main()