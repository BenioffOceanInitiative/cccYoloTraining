import json

def count_empty_annotations(coco_json_file):
    with open(coco_json_file, 'r') as file:
        data = json.load(file)

    # Extract image IDs
    image_ids = {image['id'] for image in data['images']}

    # Extract annotation image IDs
    annotation_image_ids = {annotation['image_id'] for annotation in data['annotations']}

    # Find image IDs without annotations
    images_without_annotations = image_ids - annotation_image_ids

    return [len(images_without_annotations), len(image_ids)]

# Example usage
coco_json_file = '/home/trashwheel-annotations/all_annotations.json'
data = count_empty_annotations(coco_json_file)
print(f'Number of images without annotations: {data[0]}')
print(f'Number of images overall: {data[1]}')