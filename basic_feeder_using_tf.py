import cv2
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

class VisionFeeder:
    def __init__(self, **kwargs):
        """
        init method for the VisionFeeder class.
        """
        self.image_dir = kwargs.get("image_dir", "./images")
        self.image_paths = [os.path.join(self.image_dir, filename) for filename in os.listdir(self.image_dir)]
        self.num_images = len(self.image_paths)
        
        self.resize_shape = kwargs.get("resize_shape", (256, 256))
        self.images = self.get_images(self.image_paths)
        
    def read_images(self, filepath):
        """
        Reads and preprocesses an image file for training a machine learning model.
        """
        image_tensor = tf.io.read_file(filepath)                                # Read the image file
        image_tensor = tf.image.decode_jpeg(image_tensor, channel=3)            # Decode the image file
        image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)   # Convert the image to the range [0, 1]
        image_tensor = tf.image.resize(image_tensor, self.resize_shape)         # Resize the image
        
        return image_tensor

    def get_images(self):
        """
        Returns a list of image tensors.
        """
        tensors = [self.read_images(filepath) for filepath in self.image_paths]
        return tensors
    
    def preprocess_image(self, image):
        """
        Preprocesses an image for training a machine learning model.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # Convert the image to RGB color space
        image = cv2.resize(image, self.resize_shape)        # Resize the image
        image = image / 255.0                               # Normalize the image pixel values to the range [0, 1]
        return image
    
    def train_dataset(self, batch_size):
        """
        Returns a tf.data.Dataset object that can be used to train a machine learning model.
        """
        dataset = tf.data.Dataset.from_tensor_slices(self.images)
        dataset = dataset.map(self.preprocess_image)
        # dataset = dataset.map(lambda image: tf.numpy_function(self.preprocess_image, [image], tf.float32))
        dataset = dataset.shuffle(buffer_size=self.num_images)
        dataset = dataset.batch(batch_size)
        return dataset
    
    def valid_dataset(self, batch_size):
        """
        Returns a tf.data.Dataset object that can be used to validate a machine learning model.
        """
        dataset = tf.data.Dataset.from_tensor_slices(self.images)
        dataset = dataset.batch(batch_size)
        return dataset
