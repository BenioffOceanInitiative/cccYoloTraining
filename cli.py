import os
import json
import copy
from dataset_builder import DatasetBuilder
import typing

def get_input(input_phrase)->str:
    while True:
        path = input(input_phrase)
        if path == '':
            print('You did not enter anything! Please try again.')
        elif path.lower() in ('c', 'cancel'):
            return 'c'
        elif path.lower() in ('q', 'quit'):
            exit()
        else:
            return path

def load_or_create_project()->DatasetBuilder:
    project_name = get_input("Enter the project name you would like to create or load:")
    project_dir = f'experiments/{project_name}'
    if os.path.exists(project_dir):
        print(f'Found a project with the name "{project_name}"!')
        answer = get_input("Do you want to load the project? (yes/no): ").lower()
        if answer in ('yes', 'y'):
            config = handle_change_config(project_name)
            return DatasetBuilder(project_name=project_name, annotations=config['annotations'], images=config['images'])
        elif answer in ('no', 'n'):
            pass
        else:
            print('Invalid Response')
            return load_or_create_project()
    else:
        print(f'Creating a new project with the name "{project_name}"...')
        images = get_directory('Please enter the path to the directory containing the images you would like to use: ')
        annotations = get_directory('Please enter the path to the annotations file you would like to use: ')
        print(images,annotations)
        return DatasetBuilder(project_name=project_name, annotations=annotations, images=images)

def handle_change_config(project_name:str)->dict:
    config_file = f'experiments/{project_name}/config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    data = copy.deepcopy(config)
    images = config['images']
    annotations = config['annotations']
    print(f'Loaded project {project_name} with image directory {images} and annotations {annotations}')
    new_images = get_input('Would you like to change the path to the images? (yes/no): ')
    if new_images in ('yes', 'y'):
        data['images'] = get_directory('Please enter the new path to the image directory: ')
    new_annotations = get_input('Would you like to change the path to the annotations? (yes/no): ')
    if new_annotations in ('yes', 'y'):
        data['annotations'] = get_directory('Please enter the new path to the annotations file: ')
    with open(config_file, 'w') as f:
        json.dump(data, f, indent=4)
    return data

def get_directory(text)->str:
    print(text)
    path = get_input("Enter the path to the directory: ")
    while not os.path.exists(path):
        print(f'Could not find the directory "{path}"!')
        path = get_input("Enter the path to the directory: ")
    return path

def handle_data_split()->tuple:
    split_ratio = get_input('Please enter the ratio you would like to split the dataset by train, test, and val: ex. 80/10/10 or 75/15/10 ')
    test,train,val = split_ratio.split('/')
    list_split = [test,train,val]
    test,train,val = [int(x) * 0.01 for x in list_split]
    print(test,train,val)
    if test + train + val != 1:
        print('The split ratio does not add up to 100%!')
        test,train,val = handle_data_split()
    split_seed = get_input('Please enter the seed you would like to use for the split: ')
    return test,train,val,split_seed
    
def handle_class_removal(class_dict:dict)->list:
    print("Current Classes:\n")
    for k, v in class_dict.items():
        print(f'{k}: {v}\n')
    class_ids = get_input("Enter the class ID's you would like to remove, separated by a comma. i.e. 1,2,3 or c to cancel")
    if class_ids.lower() in ['c', 'cancel']:
        return []
    class_ids = class_ids.split(',')  # Split the string into a list
    class_ids = [id.strip() for id in class_ids]  # Remove any leading or trailing whitespace
    valid_class_ids = []
    for class_id in class_ids:
        if int(class_id) not in class_dict:
            print(f'Class {class_id} does not exist!')
        elif not class_id.isdigit():
            print(f'{class_id} is not a valid class ID!')
        else:
            valid_class_ids.append(int(class_id))
    print(f'Removing classes:')
    for class_id in valid_class_ids:
        print(f'{class_dict[class_id]}\n')
    proceed = get_input('Enter p to proceed ,r to restart, or c to cancel:')
    if proceed == 'p':
        return valid_class_ids
    elif proceed == 'r':
        return handle_class_removal(class_dict)
    elif proceed == 'c':
        return []
    
def class_removal_prompt(new_dataset:DatasetBuilder)->None:
    answer = get_input('Would you like to remove classes? (y/n): ')
    if answer in ('yes', 'y'):
        class_dict = new_dataset.get_class_dict(new_dataset.annotations,False)
        print(class_dict)
        class_names = handle_class_removal(class_dict)
        if len(class_names) > 0:
            new_dataset.remove_classes(class_names)
    if answer in('n','no','c','cancel'):
        print('Skipping class removal')

def combine_datasets_prompt(new_dataset:DatasetBuilder)-> None:
    answer = get_input('Would you like to combine datasets? (y/n): ')
    if answer in ('yes', 'y'):
        annotation_file2 = get_directory('Please enter the path to the annotations file you would like to combine: ')
        new_dataset.combine_datasets(annotation_file2)
    if answer in('n','no','c','cancel'):
        print('Skipping combining datasets')

def split_data_prompt(new_dataset:DatasetBuilder)->None:
    answer = get_input('Would you like to split the dataset now? (y/n): ')
    if answer in ('yes', 'y'):
        test,train,val,seed = handle_data_split()
        new_dataset.split_data(test,train,val,seed)
    if answer in('n','no','c','cancel'):
        print('Skipping split')
         
def train_prompt(new_dataset:DatasetBuilder)->None:
    answer = get_input('Would you like to train a v8 model now? (y/n): ')
    if answer in ('yes', 'y'):
        epochs = get_input('Enter the number of epochs you would like to train for: ')
        new_dataset.train_v8_model(int(epochs))
    if answer in('n','no','c','cancel'):
        print('Skipping training')
        
def main():
    new_dataset:DatasetBuilder = load_or_create_project()
    # Combine datasets
    combine_datasets_prompt(new_dataset)
    # Remove classes
    class_removal_prompt(new_dataset)
    # Convert data
    new_dataset.to_yolo()
    # # Split data
    split_data_prompt(new_dataset)
    # Train prompt
    train_prompt(new_dataset)
    
if __name__ == "__main__":
    main()
