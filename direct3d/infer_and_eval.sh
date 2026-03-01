#!/bin/bash

source activate base
conda activate Direct3D

FILE_SUFFIX=".nii.gz"

# SLICE_AXIS=2
# PRED_DIR="/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_AeroPath/lungs_${SLICE_AXIS}/"
# GT_DIR="/PHShome/yl535/project/python/datasets/AeroPath/lungs_segmented/"
# PRED_DIR="/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_BTCV/axis_${SLICE_AXIS}/"
# GT_DIR="/PHShome/yl535/project/python/datasets/BTCV/segmented"
# PRED_DIR="/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_MSD_Brain/axis_${SLICE_AXIS}/"
# GT_DIR="/PHShome/yl535/project/python/datasets/MSD/Task01_BrainTumour/segmented/"
# PRED_DIR="/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_MSD_Liver/axis_${SLICE_AXIS}/"
# GT_DIR="/PHShome/yl535/project/python/datasets/MSD/Task03_Liver/segmented/"
# PRED_DIR="/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_MSD_Lung/axis_${SLICE_AXIS}/"
# GT_DIR="/PHShome/yl535/project/python/datasets/MSD/Task06_Lung/segmented/"

SLICE_AXIS=0

PRED_DIRS=(
  # "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_AeroPath/lungs_${SLICE_AXIS}/"
  # "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_BTCV/axis_${SLICE_AXIS}/"
  # "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_MSD_Brain/axis_${SLICE_AXIS}/"
  # "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_MSD_Liver/axis_${SLICE_AXIS}/"
  # "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_MSD_Lung/axis_${SLICE_AXIS}/"
  "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_duke_cspine/axis_${SLICE_AXIS}/"
  # "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/direct3d/results_direct3d_AbdomenCT1K/axis_${SLICE_AXIS}/"
)

GT_DIRS=(
  # "/PHShome/yl535/project/python/datasets/AeroPath/lungs_segmented/"
  # "/PHShome/yl535/project/python/datasets/BTCV/segmented"
  # "/PHShome/yl535/project/python/datasets/MSD/Task01_BrainTumour/segmented/"
  # "/PHShome/yl535/project/python/datasets/MSD/Task03_Liver/segmented/"
  # "/PHShome/yl535/project/python/datasets/MSD/Task06_Lung/segmented/"
  "/PHShome/yl535/project/python/datasets/duke_cspine/segmented/"
  # "/PHShome/yl535/project/python/datasets/AbdomenCT-1K-Image/segmented"
)

for ((i=0; i<${#PRED_DIRS[@]}; i++)); do
  PRED_DIR="${PRED_DIRS[$i]}"
  GT_DIR="${GT_DIRS[$i]}"

  python inference.py \
    --input_folder ${GT_DIR} \
    --output_folder ${PRED_DIR} \
    --slice_axis ${SLICE_AXIS}

done

# cd ../notebook/

conda deactivate
conda activate sam3d-objects

for ((i=0; i<${#PRED_DIRS[@]}; i++)); do
  PRED_DIR="${PRED_DIRS[$i]}"
  GT_DIR="${GT_DIRS[$i]}"

  python ../notebook/convert_ply_to_gif.py \
    --input_folder ${PRED_DIR} 

  python ../notebook/evaluate_shape_metrics.py \
    --pred_folder ${PRED_DIR} \
    --gt_folder ${GT_DIR} \
    --file_suffix ${FILE_SUFFIX}

done            