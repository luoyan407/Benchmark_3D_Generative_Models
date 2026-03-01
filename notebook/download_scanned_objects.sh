#!/bin/bash

INPUT_DIR="/PHShome/yl535/project/python/datasets/google_scanned_objects"

# Loop through every .zip file
for zip_file in "$INPUT_DIR"/*.zip; do
    # Skip if no zip files are found
    [ -e "$zip_file" ] || continue

    # 1. Get the filename without the path (e.g., "archive.zip")
    base_name=$(basename "$zip_file")

    # 2. Strip the .zip extension (e.g., "archive")
    dir_name="${base_name%.zip}"

    # 3. Create a specific directory for this zip file
    mkdir -p "$INPUT_DIR/$dir_name"

    echo "Extracting $base_name into $dir_name..."

    # 4. Unzip into that specific directory
    unzip -q "$zip_file" -d "$INPUT_DIR/$dir_name"
done

echo "All files organized and extracted!"