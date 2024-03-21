import argparse
import os
import json

def remove_images_and_annotations(images_directory, coco_annotations_path, search_strings):
    # Generate a list of image filenames from the directory
    images_list = [f for f in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, f))]

    # Load the COCO annotations file once
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)

    # Identify target files that contain all the search strings
    target_files = [filename for filename in images_list if all(s in filename for s in search_strings)]
    removed_files_count = 0

    # Filter out images and annotations in memory
    for target_file in target_files:
        for image in coco_data["images"]:
            if image["file_name"] == target_file:
                target_image_id = image["id"]
                # Remove the target image from the images list in the COCO data
                coco_data["images"] = [img for img in coco_data["images"] if img["id"] != target_image_id]
                # Remove any annotations referencing the target image
                coco_data["annotations"] = [anno for anno in coco_data["annotations"] if anno["image_id"] != target_image_id]
                # Delete the file
                target_file_path = os.path.join(images_directory, target_file)
                os.remove(target_file_path)
                print(f"Deleted file: {target_file_path}")
                removed_files_count += 1
                break

    # Save the updated COCO data back to the file
    if removed_files_count > 0:
        with open(coco_annotations_path, 'w') as f:
            json.dump(coco_data, f, indent=4)
        print(f"Updated COCO annotations file: {coco_annotations_path} with {removed_files_count} files removed.")
    else:
        print("No matching files found in the directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove images and their annotations based on search criteria.")
    parser.add_argument("--images-directory", type=str, default="/home/trashwheel/annotated_images", help="Path to the images directory")
    parser.add_argument("--annotations-path", type=str, default="/home/trashwheel-annotations/all_annotations.json", help="Path to the COCO annotations file")
    parser.add_argument("--search-strings", nargs='+', required=True, help="List of strings the filename must contain")

    args = parser.parse_args()

    remove_images_and_annotations(args.images_directory, args.annotations_path, args.search_strings)
