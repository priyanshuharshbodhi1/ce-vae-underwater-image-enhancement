#!/bin/bash

# Check if the main path argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <main_path> [dataset_name]"
    echo ""
    echo "Arguments:"
    echo "  main_path     - Path to the dataset folder containing train/ and val/ subdirectories"
    echo "  dataset_name  - Optional custom name for the dataset (default: LSUI)"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/dataset MyDataset"
    echo ""
    echo "Expected folder structure:"
    echo "  <main_path>/"
    echo "  ├── train/"
    echo "  │   ├── GT/      # Ground truth images"
    echo "  │   └── input/   # Degraded/input images"
    echo "  └── val/"
    echo "      ├── GT/"
    echo "      └── input/"
    exit 1
fi

# Define the main path from the argument
MAIN_PATH="$1"

# Define the dataset name (default: LSUI for backward compatibility)
DATASET_NAME="${2:-LSUI}"

# Define the output path
OUTPUT_FOLDER="./data"
mkdir -p "${OUTPUT_FOLDER}"

echo "Dataset path: ${MAIN_PATH}"
echo "Dataset name: ${DATASET_NAME}"
echo "Output folder: ${OUTPUT_FOLDER}"
echo ""

# Function to check if a file is an image
check_is_image_file() {
    local file=$1
    case "${file,,}" in
        *.png|*.jpg|*.jpeg|*.bmp|*.tiff|*.tif) return 0 ;;
        *) return 1 ;;
    esac
}

# Function to gather image files from a directory
gather_image_files() {
    local dir=$1
    find "$dir" -type f | while read -r file; do
        check_is_image_file "$file" && echo "$file"
    done | sort
}

# Function to write file paths to an output file
write_to_file() {
    local output_file=$1
    shift
    : > "$output_file" # Clear file content if it exists
    for file in "$@"; do
        echo "$file" >> "$output_file"
    done
}

# Initialize associative arrays
declare -A dataset_folder_name
declare -A output_file_name
declare -a phases=("train" "val")

# Write files to output
for phase in "${phases[@]}"; do
    echo "Processing ${phase} split..."

    # Check if directories exist
    if [ ! -d "$MAIN_PATH/${phase}/GT" ]; then
        echo "  Warning: $MAIN_PATH/${phase}/GT not found, skipping..."
        continue
    fi
    if [ ! -d "$MAIN_PATH/${phase}/input" ]; then
        echo "  Warning: $MAIN_PATH/${phase}/input not found, skipping..."
        continue
    fi

    # Gather image files
    gt_files=($(gather_image_files "$MAIN_PATH/${phase}/GT"))
    input_files=($(gather_image_files "$MAIN_PATH/${phase}/input"))

    # Write to txt using the dataset name
    write_to_file "${OUTPUT_FOLDER}/${DATASET_NAME}_${phase}_target.txt" "${gt_files[@]}"
    write_to_file "${OUTPUT_FOLDER}/${DATASET_NAME}_${phase}_input.txt" "${input_files[@]}"

    echo "  Found ${#input_files[@]} input images"
    echo "  Found ${#gt_files[@]} ground truth images"
    echo "  Created: ${DATASET_NAME}_${phase}_input.txt"
    echo "  Created: ${DATASET_NAME}_${phase}_target.txt"
done

echo ""
echo "Done! Generated files in ${OUTPUT_FOLDER}:"
ls -la "${OUTPUT_FOLDER}"/${DATASET_NAME}_*.txt 2>/dev/null || echo "No files generated."
