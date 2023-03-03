import cv2
import numpy as np
import os
import random
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from augmentation import Augmentation

class CIFAR10Feeder(Augmentation):
    def __init__(self, **kwargs):
        super().__init__()
        
        # kwargs
        self.train_batch_size = kwargs.get("train_batch_size", 32)
        self.test_batch_size = kwargs.get("train_batch_size", 32)
        self.buffer_size = kwargs.get("buffer_size", 1000)
        
        self.epochs = kwargs.get("epochs", 10)
        self.image_size = kwargs.get("image_size", (32, 32))
        self.image_dir = kwargs.get("data_dir", "./cifar-10-batches-py")


    def load_dataset(self, file, encodeType):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding=encodeType)
        return dict
    
    
    def preprocess_image(self, image):
        # Load the image file using TensorFlow
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.reshape(image, [32, 32, 3])
        return image
        
        
    def count_batch_files(self, **kwargs):
        # pattern = kwargs.get("pattern", "data_batch_")
        ##############################
        # count all train batches in directory #
        pass
    
    
    def load_batch(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


    def get_train_data(self, **kwargs):
        target_images = os.join(self.image_dir, 'data_batch_{}'.format(i))
        # load data
        train_batches = [self.load_batch(os.join(self.image_dir, 'data_batch_{}'.format(i))) for i in range(1, 6)]
        X_train = np.concatenate([batch[b'data'] for batch in train_batches], axis=0)
        y_train = np.concatenate([batch[b'labels'] for batch in train_batches], axis=0)
        X_train = X_train.reshape(-1, 3, self.image_size[0], self.image_size[1]).transpose(0, 2, 3, 1)
        
        # preprocess data
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.map(lambda x, y: (self.preprocess_image(x), y))
        train_dataset = train_dataset.shuffle(self.buffer_size).batch(self.train_batch_size).repeat(self.epochs)
        
        return train_dataset.__iter__()


    def get_valid_data(self, **kwargs):
        # load data
        test_batches = self.load_batch(os.join(self.image_dir, "test_batch"))
        X_test = np.concatenate([batch[b'data'] for batch in test_batches], axis=0)
        y_test = np.concatenate([batch[b'labels'] for batch in test_batches], axis=0)
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
    
    def augment_image(self, image, options):
        if options['crop'] is not None:
            self.random_cropping(image, (10, 10))
        
        if options['rotation'] is not None:
            self.random_rotation(image, 20)
        
        if options['flip'] is not None:
            self.random_flipping(image)
            
        if options['brightness'] is not None:
            self.random_brightness(image)
    
    def show_plots(self):
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
        for i in range(25):
            axes[i].imshow(self.X_train[i])
            axes[i].set_title(class_names[self.y_train[i]])
            axes[i].axis('off')
        plt.show()
    

    