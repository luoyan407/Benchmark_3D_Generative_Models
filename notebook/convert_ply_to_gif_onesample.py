import os
import sys
import torch
import imageio
import numpy as np
from plyfile import PlyData

# --- 1. SETUP PATHS ---
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

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
        
        # Check if 'x' exists, sometimes ply files have differnt property names
        if 'x' not in v:
            # Fallback for some ply formats (e.g. vertex_indices only) - rare but possible
            print(f"Warning: Property 'x' not found. Available: {v.properties}")
        
        points_np = np.stack([v['x'], v['y'], v['z']], axis=-1)
        num_pts = len(points_np)
        print(f"Loaded {num_pts} points from {os.path.basename(ply_path)}")

        # --- Populate Tensors ---
        # Positions: (N, 3)
        self._xyz = torch.from_numpy(points_np).float().to(device)
        
        # Scales: (N, 3)
        # Default to -4.0 (approx 0.018). 
        # If your object looks like a solid block or too thin, adjust this value.
        self._scaling = torch.full((num_pts, 3), -4.0, device=device).float()
        
        # Rotations: (N, 4) -> Identity (w=1, x=0, y=0, z=0)
        quats = torch.zeros((num_pts, 4), device=device).float()
        quats[:, 0] = 1.0
        self._rotation = quats
        
        # Opacities: (N, 1) -> High opacity
        self._opacity = torch.full((num_pts, 1), 10.0, device=device).float()
        
        # --- FIX IS HERE ---
        # Colors (Features DC): Must be (N, 1, 3) for the renderer, NOT (N, 3).
        # We start with grey (0.5).
        self._features_dc = torch.full((num_pts, 1, 3), 0.5, device=device).float()
        
        # Features Rest: (N, 0) is usually fine for empty higher-order SH
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


# --- 3. MONKEY PATCH ---
patched = False
# Search for valid class to patch in render_utils
for key in dir(render_utils):
    if "Gaussian" in key and isinstance(getattr(render_utils, key), type):
        print(f"Patching {key} with Custom Container...")
        setattr(render_utils, key, GaussianSplatContainer)
        patched = True
        break

if not patched:
    print("CRITICAL WARNING: Could not patch the Gaussian class. Script may fail.")

# --- 4. EXECUTION ---
# ply_file_path = "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/devil.ply"
ply_file_path = "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_GSO/50_BLOCKS.ply"

output_dir = os.path.dirname(ply_file_path)
base_name = os.path.splitext(os.path.basename(ply_file_path))[0]
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing scene...")
gs_object = GaussianSplatContainer(ply_file_path, device=device)

# Construct dictionary
output = {
    "gaussian": [gs_object], 
    "rotation": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0),
    "translation": torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(0),
    "scale": torch.tensor([1.0, 1.0, 1.0], device=device).unsqueeze(0)
}

print("Rendering video...")

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

save_path = os.path.join(output_dir, f"{base_name}.gif")
imageio.mimsave(save_path, video, format="GIF", duration=1000/30, loop=0)
print(f"Saved GIF to: {save_path}")