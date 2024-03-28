import os
import json
import argparse
from pathlib import Path

def load_annotations(annotations_path):
    with open(annotations_path, 'r') as file:
        data = json.load(file)
    return data

def save_annotations(data, annotations_path):
    with open(annotations_path, 'w') as file:
        json.dump(data, file, indent=4)

def find_images_with_zero_annotations(data, image_folder):
    # Image IDs with zero annotations
    zero_annotation_image_ids = {image['id'] for image in data['images'] if not any(ann['image_id'] == image['id'] for ann in data['annotations'])}
    
    # Image file paths with zero annotations and prefix 'aug_'
    image_files_to_delete = [image for image in data['images'] if image['id'] in zero_annotation_image_ids and Path(image['file_name']).name.startswith('aug_')]
    
    return image_files_to_delete

def delete_images(image_files_to_delete, image_folder):
    for image in image_files_to_delete:
        image_path = os.path.join(image_folder, image['file_name'])
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image: {image_path}")
        else:
            print(f"Image not found (already deleted?): {image_path}")

def update_annotations(data, image_files_to_delete):
    # Remove images from the annotations data
    data['images'] = [image for image in data['images'] if image not in image_files_to_delete]

    return data

def main(annotations_path, image_folder):
    # Load annotations
    data = load_annotations(annotations_path)
    
    # Find images with zero annotations and 'aug_' prefix
    image_files_to_delete = find_images_with_zero_annotations(data, image_folder)
    print(f'{len(image_files_to_delete)} images to delete.')
    
    # Delete the images
    delete_images(image_files_to_delete, image_folder)
    
    # Update annotations file
    updated_data = update_annotations(data, image_files_to_delete)
    save_annotations(updated_data, annotations_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete 'aug_' prefixed images with zero annotations and update the annotation file.")
    parser.add_argument('--annotations', default='/home/trashwheel-annotations/all_annotations.json', help='Path to the annotations file.')
    parser.add_argument('--images', default='/home/trashwheel/annotated_images', help='Path to the images folder.')
    args = parser.parse_args()

    main(args.annotations, args.images)
