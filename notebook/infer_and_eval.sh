#!/bin/bash

source activate base
conda activate benchmark_sam3d_objects

CONFIG_PATH="/PHShome/yl535/project/python/sam_3d/sam-3d-objects/checkpoints/checkpoints/pipeline.yaml"
FILE_SUFFIX=".nii.gz"

SLICE_AXIS=0

PRED_DIRS=(
  "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/results_sam3d_AeroPath/lungs_${SLICE_AXIS}/"
  "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/results_sam3d_BTCV/axis_${SLICE_AXIS}/"
  "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/results_sam3d_MSD_Brain/axis_${SLICE_AXIS}/"
  "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/results_sam3d_MSD_Liver/axis_${SLICE_AXIS}/"
  "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/results_sam3d_MSD_Lung/axis_${SLICE_AXIS}/"
  "/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/results_sam3d_duke_cspine/axis_${SLICE_AXIS}/"
)

GT_DIRS=(
  "/PHShome/yl535/project/python/datasets/AeroPath/lungs_segmented/"
  "/PHShome/yl535/project/python/datasets/BTCV/segmented"
  "/PHShome/yl535/project/python/datasets/MSD/Task01_BrainTumour/segmented/"
  "/PHShome/yl535/project/python/datasets/MSD/Task03_Liver/segmented/"
  "/PHShome/yl535/project/python/datasets/MSD/Task06_Lung/segmented/"
  "/PHShome/yl535/project/python/datasets/duke_cspine/segmented/"
)

for ((i=0; i<${#PRED_DIRS[@]}; i++)); do
  PRED_DIR="${PRED_DIRS[$i]}"
  GT_DIR="${GT_DIRS[$i]}"

  python inference_AeroPath.py \
    --config_path "${CONFIG_PATH}" \
    --input_folder "${GT_DIR}" \
    --output_folder "${PRED_DIR}" \
    --slice_axis "${SLICE_AXIS}"

  python evaluate_shape_metrics.py \
    --pred_folder "${PRED_DIR}" \
    --gt_folder "${GT_DIR}" \
    --file_suffix "${FILE_SUFFIX}"
done