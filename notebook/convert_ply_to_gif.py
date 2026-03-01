import os
import sys
import torch
import imageio
import numpy as np
from plyfile import PlyData
import argparse

# --- 1. SETUP PATHS ---
sys.path.append(os.getcwd())
# Check if parent dir exists before appending to avoid errors
parent_dir = os.path.dirname(os.getcwd())
if os.path.exists(parent_dir):
    sys.path.append(parent_dir)

# Import inference functions
from inference import (
    ready_gaussian_for_video_rendering, 
    render_video, 
    make_scene
)

# Import utils to patch
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

# --- 2. DEFINE THE CONTAINER ---
class GaussianSplatContainer:
    """
    A standalone container that holds PLY data and mimics the 
    interface expected by the renderer.
    """
    def __init__(self, ply_path, device="cuda"):
        self.device = device
        
        # Load PLY
        plydata = PlyData.read(ply_path)
        v = plydata['vertex']
        
        # Check if 'x' exists
        if 'x' not in v:
            print(f"Warning: Property 'x' not found. Available: {v.properties}")
        
        points_np = np.stack([v['x'], v['y'], v['z']], axis=-1)
        num_pts = len(points_np)
        
        # --- Populate Tensors ---
        self._xyz = torch.from_numpy(points_np).float().to(device)
        
        # Scales: Default to -4.0 (approx 0.018). 
        self._scaling = torch.full((num_pts, 3), -4.0, device=device).float()
        
        # Rotations: Identity
        quats = torch.zeros((num_pts, 4), device=device).float()
        quats[:, 0] = 1.0
        self._rotation = quats
        
        # Opacities: High opacity
        self._opacity = torch.full((num_pts, 1), 10.0, device=device).float()
        
        # Colors: Grey (0.5) with shape (N, 1, 3)
        self._features_dc = torch.full((num_pts, 1, 3), 0.5, device=device).float()
        self._features_rest = torch.zeros((num_pts, 0), device=device).float()

        # --- Metadata ---
        self.mininum_kernel_size = 0.0001
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self.sh_degree = 0 
        self.optimizer = None
        self.percent_dense = 0

    # --- Properties ---
    @property
    def get_xyz(self): return self._xyz
    @property
    def get_rotation(self): return self._rotation
    @property
    def get_scaling(self): return self._scaling
    @property
    def get_opacity(self): return self._opacity
    @property
    def get_features(self): return self._features_dc

    # --- Setters ---
    def from_xyz(self, val): self._xyz = val
    def from_rotation(self, val): self._rotation = val
    def from_scaling(self, val): self._scaling = val


# --- 3. MONKEY PATCH (Apply Once) ---
patched = False
for key in dir(render_utils):
    if "Gaussian" in key and isinstance(getattr(render_utils, key), type):
        print(f"Patching {key} with Custom Container...")
        setattr(render_utils, key, GaussianSplatContainer)
        patched = True
        break

if not patched:
    print("CRITICAL WARNING: Could not patch the Gaussian class. Script may fail.")


# --- 4. PROCESSING FUNCTION ---
def process_ply_file(ply_path, output_dir, device="cuda"):
    """
    Loads a single PLY file, renders it, and saves the GIF.
    """
    base_name = os.path.splitext(os.path.basename(ply_path))[0]
    gif_path = os.path.join(output_dir, f"{base_name}.gif")
    
    # Skip if output already exists (optional, remove check if you want to overwrite)
    if os.path.exists(gif_path):
        print(f"Skipping {base_name}, output already exists.")
        return

    print(f"Processing: {base_name}...")

    try:
        # Load object
        gs_object = GaussianSplatContainer(ply_path, device=device)

        # Construct dictionary
        output = {
            "gaussian": [gs_object], 
            "rotation": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0),
            "translation": torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(0),
            "scale": torch.tensor([1.0, 1.0, 1.0], device=device).unsqueeze(0)
        }

        # Render Pipeline
        scene_gs = make_scene(output) 
        scene_gs = ready_gaussian_for_video_rendering(scene_gs)

        video = render_video(
            scene_gs,
            r=1.5,
            fov=60,
            pitch_deg=15,
            yaw_start_deg=-45,
            resolution=512,
        )["color"]

        # Save GIF
        imageio.mimsave(gif_path, video, format="GIF", duration=1000/30, loop=0)
        print(f" -> Saved: {gif_path}")
        
        # Clean up GPU memory
        del gs_object
        del scene_gs
        del video
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error processing {base_name}: {e}")

# --- 5. MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Convert PLY files in a folder to GIFs using Sam3D renderer.")
    
    parser.add_argument(
        "--input_folder", 
        type=str, 
        required=True, 
        help="Path to the folder containing .ply files."
    )
    
    args = parser.parse_args()

    # === CONFIGURATION ===
    INPUT_FOLDER = args.input_folder # "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_MSD_Lung/axis_2"
    OUTPUT_FOLDER = INPUT_FOLDER # Save GIFs in the same folder as PLYs
    # =====================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder not found: {INPUT_FOLDER}")
        sys.exit(1)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get list of .ply files
    ply_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.ply')]
    
    if not ply_files:
        print(f"No .ply files found in {INPUT_FOLDER}")
    else:
        print(f"Found {len(ply_files)} PLY files. Starting processing...")
        
        for filename in ply_files:
            full_path = os.path.join(INPUT_FOLDER, filename)
            process_ply_file(full_path, OUTPUT_FOLDER, device)
            
        print("\nAll tasks completed.")