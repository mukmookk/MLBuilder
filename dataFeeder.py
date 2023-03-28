import cv2
import numpy as np
import os
import random
import tensorflow as tf
import pickle
from utils import *
from keras.preprocessing.image import ImageDataGenerator

from augmentation import Augmentation

class CIFAR10Feeder(Augmentation):
    def __init__(self, **kwargs):
        super().__init__()
        self.train_batch_size = kwargs.get("train_size", 32)
        self.test_batch_size = kwargs.get("test_size", 32)
        self.buffer_size = kwargs.get("buffer_size", 2000)
        
        self.epochs = kwargs.get("epochs", 10)
        self.image_size = kwargs.get("image_size", (32, 32))
        self.image_dir = kwargs.get("data_dir", "./cifar-10-batches-py")
    
    def preprocess_image(self, image):
        # Load the image file using TensorFlow
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.reshape(image, [32, 32, 3])
        return image
    
    def load_batch(self, file):
        with open(file, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            return {key.decode(): value for key, value in dict.items()}                                               # keys are byte strings, and values are regular strings. So, we need to decode keys as regular strings.


    def get_train_data(self):
        # load data
        train_batches = [self.load_batch(os.path.join(self.image_dir, 'data_batch_{}'.format(i))) for i in range(1, 6)]
        X_train = np.concatenate([batch['data'] for batch in train_batches], axis=0)                                  # extract data from batches
        y_train = np.concatenate([batch['labels'] for batch in train_batches], axis=0)                                # extract labels from batches
        X_train = X_train.reshape(-1, 3, self.image_size[0], self.image_size[1]).transpose(0, 2, 3, 1)                          # reshape data to (32, 32, 3)
        
        # preprocess data
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.map(lambda x, y: (self.preprocess_image(x), y))
        train_dataset = train_dataset.shuffle(self.buffer_size).batch(self.train_batch_size).repeat(self.epochs)
        
        return train_dataset.__iter__()


    def get_valid_data(self):
        # load data
        test_batch = self.load_batch(os.path.join(self.image_dir, "test_batch"))
        X_test = test_batch['data']
        y_test = test_batch['labels']
        X_test = X_test.reshape(-1, 3, self.image_size[0], self.image_size[1]).transpose(0, 2, 3, 1)
        
        # preprocess data
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.map(lambda x, y: (self.preprocess_image(x), y))
        test_dataset = test_dataset.batch(self.test_batch_size)
        
        return test_dataset.__iter__()
    
    
    def get_augmentation_with_ImageDataGenerator(self):
        # data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            self.image_dir,
            target_size=(32, 32),
            batch_size=self.train_batch_size,
            class_mode='categorical')
        
        validation_generator = test_datagen.flow_from_directory(
            self.image_dir,
            target_size=(32, 32),
            batch_size=self.test_batch_size,
            class_mode='categorical')
        
        return train_generator, validation_generator
    
    def augment_image(self, iter, options=None):
        if options is None:
            options = {}
            
        if options.get('crop', False):
            pil_images = self.random_cropping(iter, (10, 10))
        
        # Add more augmentation options here as needed
        
        # Convert all PIL images in the iterator to TensorFlow tensors
        image_tensors = [convert_pil_to_tensor(image) for image in pil_images]
        
        return image_tensors

###### Test ######
if __name__ == "__main__":
    feeder = CIFAR10Feeder(train_size=32, test_size=32, buffer_size=1000, epochs=10, image_size=(32, 32), data_dir="./cifar-10-batches-py")
    train_dataset = feeder.get_train_data()
    
    print_iterator_items(train_dataset)
    # feeder.show_plots()