#!/bin/bash

source activate base
conda activate sam3d-objects

FILE_SUFFIX=".nii.gz"

# SLICE_AXIS=2
# PRED_DIR="/PHShome/yl535/project/python/sam_3d/sam-3d-objects/results_AeroPath/lungs_${SLICE_AXIS}/"
# GT_DIR="/PHShome/yl535/project/python/datasets/AeroPath/lungs_segmented/"

SLICE_AXIS=2
# PRED_DIR="/PHShome/yl535/project/python/sam_3d/sam-3d-objects/results_MSD_Brain/axis_${SLICE_AXIS}/"
# GT_DIR="/PHShome/yl535/project/python/datasets/MSD/Task01_BrainTumour/segmented/"
PRED_DIR="/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_MSD_Lung/axis_${SLICE_AXIS}/"
GT_DIR="/PHShome/yl535/project/python/datasets/MSD/Task06_Lung/segmented/"


# for (( i=0; i<${#GEN_IMAGE_DIR[@]}; i++ ));               

python convert_ply_to_gif.py \
    --input_folder ${PRED_DIR} 

