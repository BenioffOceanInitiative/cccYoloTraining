import json
import numpy as np
from collections import defaultdict
import os
import sys
import shutil
from pathlib import Path
from v8train import trainv8
from utility_scripts.get_class_dict import get_class_dict
from utility_scripts.labels_per_class import labels_per_class
import sys
import random
import matplotlib.pyplot as plt

class DatasetBuilder:
    def __init__(self, project_name, annotations, images):
        self.name = project_name
        self.directory = f"experiments/{self.name}"
        self.annotations = annotations
        self.images = images
        self.config_file = f'{self.directory}/config.json'
        self.class_dict = {}

        self.new = self.check_project_exists()
        
        if not self.new:
            self.check_annotations_exists()
        
        self.prep_annotations()    
        self.check_images_exist()
                   
    def prep_annotations(self):
        '''
        checks to see if the annotations have file_name key and if the image paths are flattened, if not, it will flatten the image paths and add the file_name key and zero index the class_id's
        '''
        with open(self.annotations, 'r') as f:
            data = json.load(f)
        
        if data['categories'][0]['id'] != 0:
            print('remapping class id\'s to zero index')
            data = self.remap_classes_to_zero_index(data) 
                
        if '/' in data['images'][0]['file_name']:
            print('Flattening image paths in annotations')
            data = self.flatten_image_paths(data)
        
        if len(data['annotations']):
            if not 'file_name' in data['annotations'][0]:
                print('adding file_name key to annotations')
                data = self.add_filename_key(data)
            
        with open(self.annotations, 'w') as f:
            json.dump(data, f, indent=4)
            
    def check_project_exists(self) -> bool:
        '''
        check if the project already exists and if so , load the config file
        '''
        if not os.path.exists(self.directory) or not os.path.isdir(self.directory):
            print(f'Creating new project folder {self.directory}')
            os.mkdir(self.directory)
            class_dict = self.get_class_dict(self.annotations,False)
            labels = labels_per_class(f"{self.directory}/figures",self.annotations)
            self.class_dict = class_dict
            self.copy_annotations()
            with open(self.config_file, 'w') as f:
                json.dump({'name': self.name, 
                           'annotations': self.annotations, 
                           'images': self.images,
                           'class_dict': class_dict,
                            'labels_per_class': labels
                           }, f, indent=4)
            return True
        else:
            # load the config file
            print(f'{self.directory} already exists, attempting to load config file...')
            with open(self.config_file, 'r+') as f:
                data = json.load(f)
                for arg in data:
                    setattr(self, arg, data[arg])
                print(data)
            class_dict = self.get_class_dict(self.annotations,False)
            self.prep_annotations()
            return False
        
    def flatten_image_paths(self,data)-> dict:
        '''
        flattens the image paths in the annotations and adds file_name key 
        '''
        for i in data['images']:
            i['file_name'] = i['file_name'].split('/')[-1]
        if len(data['annotations']):
            if not 'file_name' in data['annotations'][0]:
                print('adding file_name key to annotations')
                data = self.add_filename_key(data)
            for i in data['annotations']:
                i['file_name'] = i['file_name'].split('/')[-1]
        return data
            
    def copy_images(self,annotation_file):
        '''
        copy the images to the project folder
        '''
        with open(annotation_file , 'r') as f:
            data = json.load(f)
            images = data['images']
        print(f'Copying images in annotation file to {self.directory}')
        if not os.path.exists(f'{self.directory}/images'):
            os.mkdir(f'{self.directory}/images')
            for i in images:
                try:
                    shutil.copy(f"{self.images}/{i['file_name']}", f"{self.directory}/images")
                except:
                    print(f'Error copying {i["file_name"]}, file not found')
        else:
            print(f'Images already exist in {self.directory}')
        self.images = f'{self.directory}/images'
            
    def copy_annotations(self):
        '''
        copy the annotations file to the project folder
        '''
        print(f'Copying {self.annotations} to {self.directory}')
        name = self.annotations.split('/')[-1]
        if not os.path.isfile(f'{self.directory}/{name}'):
            shutil.copy(self.annotations,f'{self.directory}/{name}')
            self.annotations = f'{self.directory}/{name}'
        else:
            print(f'Annotations file {self.annotations} already exists in {self.directory}')
            
    def check_annotations_exists(self) -> None:
        '''
        check to see if the given annoations and image directory exist, and if so, copy the annotation file to the experiment folder
        '''
        if not os.path.exists(self.annotations) or not os.path.isfile(self.annotations):
            print(f'Error: {self.annotations} does not exist, please check the path and try again, program exiting')
            sys.exit()

    def check_images_exist(self):
        '''
        check to see if the given image directory exists
        '''
        if not os.path.exists(self.images) or not os.path.isdir(self.images):
            print(f'Error: {self.images} does not exist, please check the path and try again, program exiting')
            sys.exit()
        else:
            print(f'Found image directory {self.images}')
        # check if the images have already been resized
        if os.path.exists(f"{self.name}/resized_images"):
            num_images = len(os.listdir(f"{self.directory}/resized_images"))
            print(f'Found folder with {num_images} resized images')
            self.images = f'{self.directory}/resized_images'

    def combine_datasets(self, annotation_file2)-> None:
        '''
        Reads two JSON annotation files and merges them with consistent class IDs and names.
        '''
        with open(self.annotations, 'r') as f:
            data1 = json.load(f)
        with open(annotation_file2, 'r') as f:
            data2 = self.flatten_image_paths(json.load(f))

        class_dict1 = {category['id']: category['name'] for category in data1['categories']}
        class_dict2 = {category['id']: category['name'] for category in data2['categories']}

        all_classes = set(class_dict1.values()).union(set(class_dict2.values()))
    
        class_to_new_id = {name: i for i, name in enumerate(all_classes)}

        def update_annotations(annotations, class_dict) -> list:
            for annotation in annotations:
                class_name = class_dict.get(annotation['category_id'])
                if class_name in class_to_new_id:
                    annotation['category_id'] = class_to_new_id[class_name]
            return annotations

        data1['annotations'] = update_annotations(data1['annotations'], class_dict1)
        data2['annotations'] = update_annotations(data2['annotations'], class_dict2)

        new_categories = [{"id": id, "name": name} for name, id in class_to_new_id.items()]
        
            # Remap the image ids to be unique across the two datasets
        maxn = len(data1['images'])
    
        data1_id_dict = {data1['images'][i]['id']: i +1 for i in range(maxn)}
        
        for i in range(len(data1['images'])):
            data1['images'][i]['id'] = data1_id_dict[data1['images'][i]['id']]
        
        for i in range(len(data1['annotations'])):
            data1['annotations'][i]['id'] = i
            data1['annotations'][i]['image_id'] = data1_id_dict[data1['annotations'][i]['image_id']]
                
        data2_id_dict = {data2['images'][i]['id']: 1 + i + maxn for i in range(len(data2['images']))}
        
        for i in range(len(data2['images'])):
            data2['images'][i]['id'] = data2_id_dict[data2['images'][i]['id']]
            
        for i in range(len(data2['annotations'])):
            data2['annotations'][i]['id'] = 1 + i + maxn
            data2['annotations'][i]['image_id'] = data2_id_dict[data2['annotations'][i]['image_id']] 
             
        combined_annotations = data1['annotations'] + data2['annotations']
        combined_images = data1['images'] + data2['images']
        combined_data = {
            'images': combined_images,
            'annotations': combined_annotations,
            'categories': new_categories
        }

        file_name = f'{self.directory}/{self.name}_combined.json'
        with open(file_name, 'w') as f:
            json.dump(combined_data, f, indent=4)
        print(f"Combined annotations saved to {file_name}")
        class_dict = get_class_dict(file_name,False)
        os.makedirs(f'{self.directory}/figures', exist_ok=True)
        labels=labels_per_class(f"{self.directory}/figures",file_name)
        self.update_config(annotations=file_name,class_dict=class_dict,labels_per_class=labels,combined_with=annotation_file2,original_annotations=self.annotations)
        self.annotations = file_name
        self.class_dict = class_dict
        self.labels_per_class = labels

    def update_config(self,**kwargs)-> None:
        '''
        Updates the config file for the project.
        '''
        with open(f'{self.directory}/config.json', 'r+') as f:
            data = json.load(f)
            for arg in kwargs:
                data[arg] = kwargs[arg]
                self.arg = kwargs[arg]
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

    def add_filename_key(self,data) -> dict:
        '''
        Adds filename key to annotations file if missing
        '''
        image_dict = {}
       
        for i in data['images']:
            image_dict[i['id']] = i['file_name']
        for a in data['annotations']:
            if not 'file_name' in a:
                a['file_name'] = image_dict[a['image_id']]
        return data

    def remove_classes(self,classes_to_remove) -> None:
        """
        Remove classes from annotations file remaps class ids, and creates a yaml file for yolo training
        :param classes_to_remove: List of classes to remove
        """
        output_file = f'{self.directory}/{self.name}_removed.json'
        with open(self.annotations, 'r') as f:
            annotations = json.load(f)  
        class_dict = {}

        for i in  annotations['categories']:
            class_dict[i['id']] = i['name']
        
        print("Classes to remove:")
        print(class_dict)

        for i in classes_to_remove:
            print(class_dict[i])

        annotations['annotations'] = [c for c in annotations['annotations'] if c['category_id'] not in classes_to_remove]
        annotations['categories'] = [c for c in annotations['categories'] if c['id'] not in classes_to_remove]

        class_dict.clear()
        # get new class dict
        for i in  annotations['categories']:
            class_dict[i['id']] = i['name']
        # Create a mapping from class_name to a new id
        class_to_new_id = {class_name: new_id for new_id, class_name in enumerate(class_dict.values())}
        #print(class_to_new_id)
        # Update the 'id' in categories using the new mapping
        for category in annotations['categories']:
            category['id'] = class_to_new_id[category['name']]
        # Update the 'category_id' in annotations using the new mapping
        for annotation in annotations['annotations']:
            original_class_name = class_dict[annotation['category_id']]
            annotation['category_id'] = class_to_new_id[original_class_name] 

        with open(output_file, 'w') as f:
            json.dump(annotations, f)
        class_dict = get_class_dict(output_file,False)
        label_dict = labels_per_class(f"{self.directory}/figures",output_file)
        self.update_config(annotations = output_file,class_dict=class_dict,labels_per_class=label_dict)
        self.annotations = output_file
        self.class_dict = class_dict
        print(f"Annotations saved to {output_file}")
        os.makedirs(f'{self.directory}/figures', exist_ok=True)
        with open(f'{self.directory}/figures/{self.name}_class_dict.json', 'w') as f:
            json.dump(class_dict, f, indent=4)

    def remap_classes_to_zero_index(self,data)->dict:
        '''
        given an annotation dictionary, remaps the class_id's to 0 indexed and fixes missing class_id's.
        returns the updated annotation dictionary
        '''
        class_dict = {}
        for i in  data['categories']:
            class_dict[i['id']] = i['name']

        class_to_new_id = {class_name: new_id for new_id, class_name in enumerate(class_dict.values())}

        for category in data['categories']:
            category['id'] = class_to_new_id[category['name']]

        for annotation in data['annotations']:
        
            original_class_name = class_dict[annotation['category_id']]
            annotation['category_id'] = class_to_new_id[original_class_name]
       
        self.class_dict = class_dict
        return data
    
    def write_json_file(self,data, file_name):
        '''
        writes a json file to the given file name
        '''
        with open(file_name, 'r+') as f:
            json.dump(data, f, indent=4)

    def data_yaml_file(self):
        '''
        Creates a data.yaml file for use with YOLOv5 and v8.
        '''
        with open(self.annotations, 'r') as f:
            data = json.load(f)
        with open(f"{self.directory}/data/datav8.yaml","w") as f:
            f.write("train: train/images\n")
            f.write("test: test/images \n")
            f.write("val: val/images\n")
            f.write(f"nc: {len(self.class_dict)}\n")
            f.write(f"names: {list(self.class_dict.values())}")
        print(f'Created data.yaml file for {self.name}')
        with open(f"{self.directory}/data/datav5.yaml","w") as f:
            f.write(f"train: {self.name}/train/images\n")
            f.write(f"test: {self.name}/test/images \n")
            f.write(f"val: {self.name}val/images\n")
            f.write(f"nc: {len(self.class_dict)}\n")
            f.write(f"names: {list(self.class_dict.values())}")
        print(f'Created data.yaml file for {self.name}')

    def to_yolo(self):
        '''
        Converts COCO annotations to YOLO format, compatible with v5 and v8.
        '''
        counter = 0
    
        with open(self.annotations) as f:
            fn = Path(self.directory) / 'labels'   # folder name
            fn.mkdir(exist_ok=True, parents=True)  # make folder
            data = json.load(f)
            print('Number of images:', len(data['images']))
            print('Number of annotations:', len(data['annotations']))
            print(f'Converting {self.annotations}...')
            # Create image dict
            #images = {x['id']: x for x in data['images']}
            images = {}
            for x in data['images']:
                
                if x['id'] not in images:
                    images[x['id']] = x
                else:
                    counter += 1
                    #sys.stdout.write(f'\r {counter} duplicate images found')  
            for i in range(len(data['images'])):
                with open((fn / data['images'][i]['file_name']).with_suffix('.txt'), 'w') as file:
                    pass
            # Create image-annotations dict
            imgToAnns = defaultdict(list)
            ann_counter = 0
            for ann in data['annotations']:
                ann_counter += 1
                imgToAnns[ann['image_id']].append(ann)
        
            print(f'Number of duplicate images: {counter}')
            # Write labels file
            for img_id, anns in imgToAnns.items():
                img = images[img_id]
                h, w, f = img['height'], img['width'], img['file_name']
                bboxes = []
                for ann in anns:
                    # The COCO box format is [top left x, top left y, width, height]
                    box = np.array(ann['bbox'], dtype=np.float64)
                    box[:2] += box[2:] / 2  # xy top-left corner to center
                    box[[0, 2]] /= w  # normalize x
                    box[[1, 3]] /= h  # normalize y
                    if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                        continue
                    cls = ann['category_id']
                    box = [cls] + box.tolist()
                    if box not in bboxes:
                        bboxes.append(box)
                # Write
                with open((fn / f).with_suffix('.txt'), 'a') as file:
             
                    for i in range(len(bboxes)):
                        line = *(bboxes[i]),  # cls, box or segments
                        file.write(('%g ' * len(line)).rstrip() % line + '\n')
        
    def split_indices(self,x, train, test, validate):
        '''
        splits the indices of the data into train, test, and validation sets.
        '''
        n = len(x)
        v = np.arange(n)
        i = round(n * train)  # train
        j = round(n * test) + i  # test
        k = round(n * validate) + j  # validate
        return v[:i], v[i:j], v[j:k]  # return indices
    
    def split_data(self, train, test, validate,seed):
        '''
        Splits the data into train, test, and validation sets and moves the images to the proper data sets.
        provide the ratios of train test and validate in decimal percentage.
        provide a seed for the random shuffle for reproducibility (optional).
        '''
        err_counter = 0
        output_dir = f'{self.directory}/data'
        files = os.listdir(self.directory + '/labels')
        if files == []:
            print('No label files found, please check the path and try again')
            sys.exit()

        if seed == None:
            seed = 42
        random.Random(seed).shuffle(files)
        self.update_config(annotations=self.annotations, images=self.images, seed=seed)
        i, j, k = self.split_indices(files, train, test, validate)
        datasets = {'train': i, 'test': j, 'val': k}
        print('train: ',len(datasets['train']),'test: ',len(datasets['test']), 'val: ', len(datasets['val']))
        for dataset in datasets:
            self.make_dirs(f'{output_dir}/{dataset}')
            directory = f'{output_dir}/{dataset}/labels'
            
            for index in datasets[dataset]:
                image_name = files[index].split('.')[0] + '.' + 'jpg'
                try:
                    shutil.copy(f'{self.images}/{image_name}', f'{output_dir}/{dataset}/images')
                    shutil.copy(f'{self.directory}/labels/{files[index]}', directory)
                except:
                    print(f'Error copying {files[index]} with error {sys.exc_info()}')
                    err_counter += 1
                    continue
        print('done with ', err_counter, 'errors')
        for key in datasets:
            self.output_num_labels(f'{output_dir}/{key}/labels')  
        self.data_yaml_file()
    
    def output_num_labels(self,file_dir):
        """
        outputs the number of labels per class in data set.
        """
        files = os.listdir(file_dir)
        label_dict = {}
        for file in files:
            
            with open(f'{file_dir}/{file}', 'r') as f:
                lines = [line.rstrip('\n').split() for line in f]
                for line in lines:
                    if(len(line) == 0) or len(line[0]) > 2:
                        continue
                    if int(line[0]) not in label_dict:
                        label_dict[int(line[0])] = 1
                    else:
                        label_dict[int(line[0])] += 1
        return label_dict

    def get_class_dict(self,annotation_file,write:bool)->dict:
        '''
        reads a json annotation file and returns a dictionary of class_id's and class_names
        optionally writes the dictionary to a json file with the name <annotation_file>_dict.json
        '''
        class_dict = {}
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        for i in  data['categories']:
            class_dict[i['id']] = i['name']
        #print(class_dict)
        #split file name and extension
        if write:
            annotation_file = annotation_file.split('/',1)[-1]
            annotation_file = annotation_file.split('.',1)[0]
            annotation_file +='_dict.json'
            with open(annotation_file, 'w') as f:
                json.dump(class_dict, f, indent=4)
            print(f'Class dict written to {annotation_file}')
        return class_dict
    
    def train_v8_model(self,epochs):
        '''
        trains yolov8 on the dataset
        '''
        trainv8(f'{self.directory}/data/datav8.yaml',epochs)
        
    def make_dirs(self,dir='new_dir/'):
        # Create folders
        dir = Path(dir)
        if dir.exists():
            shutil.rmtree(dir)  # delete dir
        for p in dir, dir / 'labels', dir / 'images':
            p.mkdir(parents=True, exist_ok=True)  # make dir
        return dir