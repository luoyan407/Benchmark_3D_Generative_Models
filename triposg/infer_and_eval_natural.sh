#!/bin/bash

source activate base
conda activate triposg

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

SLICE_AXIS=1

PRED_DIRS=(
  "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/triposg/results_triposg_GSO/"
  "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/triposg/results_triposg_Animal3D/"
)

GT_DIRS=(
  "/PHShome/yl535/project/python/datasets/google_scanned_objects/"
  "/PHShome/yl535/project/python/datasets/animal3d/"
)

GT_OBJ_DIRS=(
  "/PHShome/yl535/project/python/datasets/google_scanned_objects_preprocessed/"
  "/PHShome/yl535/project/python/datasets/animal3d/preprocessed"
)

# for ((i=0; i<${#PRED_DIRS[@]}; i++)); do
#   PRED_DIR="${PRED_DIRS[$i]}"
#   GT_DIR="${GT_DIRS[$i]}"
#   GT_OBJ_DIR="${GT_OBJ_DIRS[$i]}"

#   # if (( i == 0 )); then

#   python scripts/inference_GSO.py \
#     --input_folder ${GT_DIR} \
#     --output_folder ${PRED_DIR} 

#   # python scripts/inference_Animal3D.py \
#   #   --input_folder ${GT_DIR} \
#   #   --output_folder ${PRED_DIR} 

# done

conda deactivate
conda activate sam3d-objects

for ((i=0; i<${#PRED_DIRS[@]}; i++)); do
  PRED_DIR="${PRED_DIRS[$i]}"
  GT_DIR="${GT_DIRS[$i]}"
  GT_OBJ_DIR="${GT_OBJ_DIRS[$i]}"

  python ../notebook/convert_ply_to_gif.py \
    --input_folder ${PRED_DIR} 

  python ../notebook/evaluate_shape_metrics_natural.py \
    --pred_folder ${PRED_DIR} \
    --gt_folder ${GT_OBJ_DIR} \
    --file_suffix ${FILE_SUFFIX}

done            