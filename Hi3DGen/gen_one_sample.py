import os
import argparse
import torch
import numpy as np
import trimesh
from PIL import Image
from hi3dgen.pipelines import Hi3DGenPipeline
from huggingface_hub import snapshot_download

# --- Configuration ---
os.environ['SPCONV_ALGO'] = 'native'
MAX_SEED = np.iinfo(np.int32).max
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')

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

def main():
    parser = argparse.ArgumentParser(description="Hi3DGen CLI: Single Image to 3D Mesh")
    
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="output.ply", help="Path to save output mesh (e.g., output.glb, output.obj)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    
    # Advanced Generation Settings
    parser.add_argument("--ss-steps", type=int, default=50, help="Sparse Structure Sampling Steps")
    parser.add_argument("--ss-strength", type=float, default=3.0, help="Sparse Structure Guidance Strength")
    parser.add_argument("--slat-steps", type=int, default=6, help="Structured Latent Sampling Steps")
    parser.add_argument("--slat-strength", type=float, default=3.0, help="Structured Latent Guidance Strength")
    
    args = parser.parse_args()

    # 1. Ensure weights are present
    cache_weights(WEIGHTS_DIR)

    # 2. Load Models
    pipeline, normal_predictor = load_models()

    # 3. Process
    process_single_image(
        args.input,
        args.output,
        pipeline,
        normal_predictor,
        seed=args.seed,
        ss_steps=args.ss_steps,
        ss_strength=args.ss_strength,
        slat_steps=args.slat_steps,
        slat_strength=args.slat_strength
    )

if __name__ == "__main__":
    main()