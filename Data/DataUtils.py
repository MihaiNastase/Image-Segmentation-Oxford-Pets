import os # To access system files
from IPython.display import display # Utility to easily display images in this environment
# For image manipulation
import PIL 
from PIL import Image, ImageOps

import numpy as np # To transform and manipulate image data

import random # For testing and evaluation

# Required Deep Learning libraries
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img # Utility to load images into tensors

""" Function to collect the paths of all image data """
def get_image_paths(input_dir, target_dir):
    input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    return input_img_paths, target_img_paths

""" Function to split image data between train and validaion sets """
def train_test_split(input_img_paths, target_img_paths, split=0.8, r=0):
    """ Input same random seed to ensure that the the image paths are index aligned """
    random.Random(r).shuffle(input_img_paths)
    random.Random(r).shuffle(target_img_paths)
    """ Do split based on percentage """
    train_input_img_paths = input_img_paths[:int(len(input_img_paths)*split)]
    train_target_img_paths = target_img_paths[:int(len(target_img_paths)*split)]
    test_input_img_paths = input_img_paths[-int(len(input_img_paths)*round(1-split,1)):]
    test_target_img_paths = target_img_paths[-int(len(target_img_paths)*round(1-split,1)):]
    
    return train_input_img_paths, train_target_img_paths, test_input_img_paths, test_target_img_paths

""" Handler to iterate over the data (as Numpy arrays) """
class DataHandler(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    """ Returns tuple (input, target) correspond to batch #idx. """
    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            y[j] -= 1  # Ground truth labels are 1, 2, 3. We subtract one to make them 0, 1, 2:
        return x, y