import os
import cv2
import glob
import numpy as np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--input_image_folder", type=str, required=True)
    parser.add_argument("--input_mask_folder", type=str, required=True)
    parser.add_argument("--output_mask_folder", type=str, required=True)
    parser.add_argument("--output_masked_images_folder", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.data_dir, args.output_mask_folder), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, args.output_masked_images_folder), exist_ok=True)
    for fn in glob.glob(f"{args.data_dir}/{args.input_mask_folder}/*.png"):
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        # _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 1. Use smaller kernel for more precise morphology 3 instead of 5
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # 2. Remove noise
        thresh = cv2.dilate(thresh, kernel, iterations=1)         # 3. Expand slightly
        thresh = cv2.erode(thresh, kernel, iterations=2)          # 4. Shrink more to break weak connections

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4) # 5. Use stricter connectivity 4 instead of 8
        
        # 6. Filter out small components - Add area filtering to avoid tiny components
        # Calculate min area based on largest component
        largest_area = np.max(stats[1:, cv2.CC_STAT_AREA])
        min_area = int(largest_area * 0.1)  # 10% of largest component
        # For 1080p video (1920x1080):
        # - Full frame = 2,073,600 pixels
        # - Typical human figure might be ~20% = ~400,000 pixels
        # - Small fragments might be 1-2% = 4,000-8,000 pixels
        min_area = max(500, min(min_area, 5000))

        # Use calculated min_area in component filtering
        valid_labels = np.where(stats[1:, cv2.CC_STAT_AREA] > min_area)[0] + 1
        if len(valid_labels) > 0:
            largest_component_index = valid_labels[np.argmax(stats[valid_labels, cv2.CC_STAT_AREA])]
        else:
            largest_component_index = 0
        largest_component_mask = (labels == largest_component_index).astype(np.uint8) * 255

        cv2.imwrite(fn.replace(args.input_mask_folder, args.output_mask_folder), largest_component_mask)

        mask = largest_component_mask > 0
        img = cv2.imread(fn.replace(args.input_mask_folder, args.input_image_folder))
        img[~mask] = 0
        cv2.imwrite(fn.replace(args.input_mask_folder, args.output_masked_images_folder), img)
