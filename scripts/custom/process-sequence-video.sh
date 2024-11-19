#!/bin/bash

# Check if path is specified as an argument
if [ -z "$1" ]; then
  echo "Please specify the path to the folder containing the video."
  exit 1
fi

# Check if the path exists
if [ ! -d "$1" ]; then
  echo "The specified path does not exist or is not a directory."
  exit 1
fi

# Check if path is specified as an argument
if [ -z "$2" ]; then
  echo "Please specify the name of the video."
  exit 1
fi

# Check if the path exists
if [ ! -f "$1/$2" ]; then
  echo "The specified path does not exist or is not a file."
  exit 1
fi

# Check if gender is specified as an argument
if [ -z "$3" ]; then
  echo "Please specify the gender."
  exit 1
fi

# Add after the existing parameter checks
FORCE_RERUN=false
if [ "$4" = "--force_rerun" ]; then
  FORCE_RERUN=true
  echo "Force rerun is set to true."
fi

VIDEO_FOLDER=$(readlink -f "$1")
VIDEO_NAME=$2
GENDER=$3

RAW_IMAGE_FOLDER="raw_images"
PREPROCESSED_IMAGE_FOLDER="preprocessed_images"
INPUT_IMAGE_FOLDER=$RAW_IMAGE_FOLDER

# Preprocess video frames
if [ ! -d "$VIDEO_FOLDER/$PREPROCESSED_IMAGE_FOLDER" ] || [ "$FORCE_RERUN" = true ]; then
  python scripts/custom/video-utils.py \
    --video_folder $VIDEO_FOLDER \
    --video_name $VIDEO_NAME \
    --output_folder $RAW_IMAGE_FOLDER
  echo "converted video $VIDEO_FOLDER/$VIDEO_NAME to images in $VIDEO_FOLDER/$RAW_IMAGE_FOLDER"

  python scripts/custom/preprocess_image.py \
    --data_dir $VIDEO_FOLDER/$RAW_IMAGE_FOLDER \
    --image_processed_dir $VIDEO_FOLDER/$PREPROCESSED_IMAGE_FOLDER
  echo "Preprocessed images in $VIDEO_FOLDER/$PREPROCESSED_IMAGE_FOLDER"
else
  echo "Preprocessed images folder exists. Skipping."
fi

# Run OpenPose
if [ ! -f "$VIDEO_FOLDER/keypoints.npy" ] || [ "$FORCE_RERUN" = true ]; then
  echo "Running OpenPose for $VIDEO_FOLDER/$INPUT_IMAGE_FOLDER"
  bash scripts/custom/run-openpose-bin.sh $VIDEO_FOLDER/$INPUT_IMAGE_FOLDER
  echo "OpenPose output in $VIDEO_FOLDER/openpose_json"
else
  echo "OpenPose output exists. Skipping."
fi

# Run SAM
if [ ! -d "$VIDEO_FOLDER/$MASK_FOLDER" ] || [ "$FORCE_RERUN" = true ]; then
  echo "Running mask in $VIDEO_FOLDER"
  python scripts/custom/run-sam.py \
    --data_dir $VIDEO_FOLDER \
    --image_folder $INPUT_IMAGE_FOLDER
  # python scripts/custom/run-rvm.py --data_dir $VIDEO_FOLDER

  # detect head segmentation with SCHP
  python scripts/custom/run-schp.py --data_dir $VIDEO_FOLDER --image_folder $INPUT_IMAGE_FOLDER

  INPUT_MASK_FOLDER="body_only_masks_schp"
  OUTPUT_MASK_FOLDER="masks"
  OUTPUT_MASKED_IMAGES_FOLDER="masked_images"
  python scripts/custom/extract-largest-connected-components.py \
    --data_dir $VIDEO_FOLDER \
    --input_image_folder $INPUT_IMAGE_FOLDER \
    --input_mask_folder $INPUT_MASK_FOLDER \
    --output_mask_folder $OUTPUT_MASK_FOLDER \
    --output_masked_images_folder $OUTPUT_MASKED_IMAGES_FOLDER
else
  echo "Masks folder exists. Skipping."
fi

MASKED_IMAGES_FOLDER=$OUTPUT_MASKED_IMAGES_FOLDER
# Run ROMP
if [ ! -f "$VIDEO_FOLDER/poses.npz" ] || [ "$FORCE_RERUN" = true ]; then
  python scripts/custom/run-romp.py --data_dir $VIDEO_FOLDER --image_folder $INPUT_IMAGE_FOLDER
  # python scripts/custom/run-romp.py \
  #   --data_dir $VIDEO_FOLDER \
  #   --image_folder $MASKED_IMAGES_FOLDER

  python scripts/visualize-SMPL.py \
    --path $VIDEO_FOLDER \
    --gender $GENDER \
    --pose $VIDEO_FOLDER/poses.npz \
    --headless \
    --fps 10 \
    --image_folder $INPUT_IMAGE_FOLDER \
    --skeleton_only
else
  echo "ROMP output exists. Skipping."
fi

# Run SMPL
if [ ! -f "$VIDEO_FOLDER/poses_optimized.npz" ] || [ "$FORCE_RERUN" = true ]; then
  echo "Refining SMPL..."
  # python scripts/custom/refine-smpl.py --data_dir $VIDEO_FOLDER --gender $GENDER --silhouette
  python scripts/custom/refine-smpl-large-errors.py \
    --data_dir $VIDEO_FOLDER \
    --gender $GENDER \
    --silhouette
else
  echo "SMPL output exists. Skipping."
fi

# Run visualization
if [ ! -f "$VIDEO_FOLDER/output.mp4" ] || [ "$FORCE_RERUN" = true ]; then
  python scripts/visualize-SMPL.py \
    --path $VIDEO_FOLDER \
    --gender $GENDER \
    --pose $VIDEO_FOLDER/poses_optimized.npz \
    --headless \
    --fps 10 \
    --image_folder $INPUT_IMAGE_FOLDER
else
  echo "Output video exists. Skipping."
fi
