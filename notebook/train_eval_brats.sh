#!/bin/bash


conda activate sam3d-objects  # CORRECT NAME!

# Correct checkpoint path for your machine
CONFIG_PATH="/shared/ssd_14T/home/serenaliu/benchmark_sam3d/checkpoints/hf/pipeline.yaml"
FILE_SUFFIX=".nii.gz"

# Your BraTS data paths
SLICE_AXIS=2  # axial slices (standard for brain imaging)
PRED_DIR="/shared/ssd_14T/home/serenaliu/sam3d_results/brats_predictions/axis_${SLICE_AXIS}/"
GT_DIR="/shared/ssd_14T/home/serenaliu/BRATSPKG/brats_segmented/"

echo "=========================================="
echo "BraTS SAM3D Pipeline"
echo "=========================================="
echo "Config:  ${CONFIG_PATH}"
echo "Input:   ${GT_DIR}"
echo "Output:  ${PRED_DIR}"
echo "Axis:    ${SLICE_AXIS}"
echo "=========================================="

# Run SAM3D inference
python inference_BraTS.py \
    --config_path "${CONFIG_PATH}" \
    --input_folder "${GT_DIR}" \
    --output_folder "${PRED_DIR}" \
    --slice_axis ${SLICE_AXIS}

# Evaluate result
python evaluate_shape_metrics.py \
    --pred_folder "${PRED_DIR}" \
    --gt_folder "${GT_DIR}" \
    --file_suffix "${FILE_SUFFIX}"