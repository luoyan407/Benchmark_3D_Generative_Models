#!/bin/bash

source activate base
conda activate benchmark_sam3d_objects

CONFIG_PATH="/PHShome/yl535/project/python/sam_3d/sam-3d-objects/checkpoints/checkpoints/pipeline.yaml"

PRED_DIR=/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/results_sam3d_GSO/
GT_DIR=/PHShome/yl535/project/python/datasets/google_scanned_objects/
GT_OBJ_DIR=/PHShome/yl535/project/python/datasets/google_scanned_objects_preprocessed

python inference_GSO_time.py \
  --config_path "${CONFIG_PATH}" \
  --input_folder "${GT_DIR}" \
  --output_folder "${PRED_DIR}"

python evaluate_shape_metrics_natural.py \
    --pred_folder "${PRED_DIR}" \
    --gt_folder "${GT_OBJ_DIR}" \
    --file_suffix "${FILE_SUFFIX}"

PRED_DIR=/PHShome/yl535/project/python/sam_3d/benchmark_sam3d/results_sam3d_Animal3D/
GT_DIR=/PHShome/yl535/project/python/datasets/animal3d/
GT_OBJ_DIR=/PHShome/yl535/project/python/datasets/animal3d/preprocessed

python inference_Animal3D.py \
  --config_path "${CONFIG_PATH}" \
  --input_folder "${GT_DIR}" \
  --output_folder "${PRED_DIR}"

python evaluate_shape_metrics_natural.py \
    --pred_folder "${PRED_DIR}" \
    --gt_folder "${GT_OBJ_DIR}" \
    --file_suffix "${FILE_SUFFIX}"
