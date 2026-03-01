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

import os
import imageio
import uuid
# from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer

import argparse
import os
import sys
from glob import glob
from typing import Any, Union

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from image_process import prepare_image
from briarmbg import BriaRMBG

import pymeshlab

def get_all_directories_pathlib(input_folder):
    path = Path(input_folder)
    # .rglob('*') recursively finds everything; we filter for directories
    directories = [str(f) for f in path.rglob('*') if f.is_dir()]
    return directories

@torch.no_grad()
def run_triposg(
    pipe: Any,
    image_input: Union[str, Image.Image],
    rmbg_net: Any,
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    faces: int = -1,
) -> trimesh.Scene:

    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)

    outputs = pipe(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).samples[0]
    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))

    if faces > 0:
        mesh = simplify_mesh(mesh, faces)

    return mesh

def mesh_to_pymesh(vertices, faces):
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    return ms

def pymesh_to_trimesh(mesh):
    verts = mesh.vertex_matrix()#.tolist()
    faces = mesh.face_matrix()#.tolist()
    return trimesh.Trimesh(vertices=verts, faces=faces)  #, vID, fID

def simplify_mesh(mesh: trimesh.Trimesh, n_faces):
    if mesh.faces.shape[0] > n_faces:
        ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum = n_faces)
        return pymesh_to_trimesh(ms.current_mesh())
    else:
        return mesh

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
    
    # --- FIX 1: Handle 4D masks (just like you did for images) ---
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
    parser.add_argument(
        '--slice_axis', 
        type=int, 
        default=2,
        help='Path to the folder to save output results.'
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--faces", type=int, default=-1)
    
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16

    # download pretrained weights
    triposg_weights_dir = "pretrained_weights/TripoSG"
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

    # init rmbg model for background removal
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval() 

    # init tripoSG pipeline
    pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(triposg_weights_dir).to(device, dtype)


    input_folder = args.input_folder # f"/PHShome/yl535/project/python/datasets/AeroPath/lungs_segmented" 
    output_dir = args.output_folder  # f"/PHShome/yl535/project/python/sam_3d/sam-3d-objects/results_AeroPath/lungs" 

    # create output directory if it doesn't exist
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

    # pairs = find_image_mask_pairs(input_folder)
    # for scan_path,mask_path  in pairs:
    #     print("IMG :", scan_path.name)
    #     print("MASK:", mask_path.name)
    #     objs = extract_all_objects_middle_slices(scan_path, mask_path, axis=args.slice_axis)
    #     extracted_scan, extracted_mask = objs[0][0], objs[0][1]

        # # Guard clause: Check if the extracted scan is empty/black
        # if np.max(extracted_scan) == 0:
        #     print(f"Skipping {scan_path.name}: Extracted slice is empty.")
        #     continue

        # output_file = f"{output_dir}/{scan_path.name}.ply"

        # extracted_scan_pil = Image.fromarray(extracted_scan)

        # # 1. Save the PIL image to a temporary path on disk
        # temp_img_path = os.path.join(output_dir, f"temp_{scan_path.name}.png")
        # extracted_scan_pil.save(temp_img_path)

        # 2. Run TripoSG with error handling
        try:
            run_triposg(
                pipe,
                image_input=file_path, 
                rmbg_net=rmbg_net,
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                faces=args.faces,
            ).export(output_file)
            print(f"Saved mesh to {output_file}")
            
        except ValueError as e:
            if "max() arg is an empty sequence" in str(e):
                print(f"Skipping {file_path}: Background removal resulted in empty image.")
            else:
                print(f"Error processing {file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error on {file_path}: {e}")

        # 3. (Optional) Cleanup the temporary file
        # if os.path.exists(temp_img_path):
        #     os.remove(temp_img_path)

