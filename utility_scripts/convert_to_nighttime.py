import os
import sys
import albumentations as A
import cv2
import matplotlib as plt
import numpy as np

def convert_image_to_nighttime(image_path, output_folder):
    image_location = '/'.join(image_path.split('/')[:-1])
    output_path = output_folder + '/' + image_path.split('/')[-1][:-4] + '_nighttime.jpg'
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # Turns image to grayscale and adds noise
        
    transform = A.Compose([
        A.ToGray(1),
        A.ISONoise(),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, -0.2), contrast_limit=(-0.3, -0.3), p=1),     
    ])
    
    image = cv2.imread(image_path)

    #Creates vignette effect
    rows, cols = image.shape[:2]

    X_resultant_kernel = cv2.getGaussianKernel(cols,200)
    Y_resultant_kernel = cv2.getGaussianKernel(rows,200)

    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
    output = np.copy(image)

    for i in range(3):
        output[:,:,i] = output[:,:,i] * mask
    
    transformed = transform(image=output)["image"]
    cv2.imwrite(output_path, transformed)
        
    
image_path = "C:/Users/legop/Desktop/GitHub/cccYoloTraining/utility_scripts/daytime_images/image.png"
output_folder = "C:/Users/legop/Desktop/GitHub/cccYoloTraining/utility_scripts/transformed_images"

convert_image_to_nighttime(image_path, output_folder)