import tensorflow as tf
import random
import numpy as np
from PIL import Image

class Augmentation:
    def __init__(self):
        pass
    
    def crop_image(self, iterator, coordinates=(0, 0, 32, 32)):
        cropped_images = []  # Change the variable name to avoid conflicts
        for batch in iterator:
            for image in batch[0]:  # Assuming that the first element of the tuple is the batch of images
                # Convert data type to uint8
                image_uint8 = image.numpy().astype(np.uint8)

                cropped_image = Image.fromarray(image_uint8).crop(coordinates)
                cropped_images.append(cropped_image)  # Append the cropped_image to the list

        return cropped_images

    def random_cropping(self, iter, cropped_size):
        """
        Augmentation option #1
        Randomly crop a PIL Image to the given size.
        Args:
            image (PIL Image): Image to be cropped.
            size (tuple): Size of the crop region (width, height).
        Returns:
            PIL Image: Cropped image.
        """
        width, height = (32, 32)
        left = random.randint(0, width - cropped_size[0])
        top = random.randint(0, height - cropped_size[1])
        right = left + cropped_size[0]
        bottom = top + cropped_size[1]
        
        return self.crop_image(iter, (left, top, right, bottom))
    

    def random_rotation(self, image, max_angle):
        """
        Randomly rotate a PIL Image by a maximum angle.
        Args:
            image (PIL Image): Image to be rotated.
            max_angle (float): Maximum rotation angle in degrees.
        Returns:
            PIL Image: Rotated image.
        """
        angle = random.uniform(-max_angle, max_angle)
        return image.rotate(angle)
    
    
    def random_flipping(self, image):
        """
        Randomly flip a TensorFlow tensor image horizontally or vertically.
        Args:
            image (Tensor): Image to be flipped.
        Returns:
            Tensor: Flipped image.
        """
        return tf.image.random_flip_left_right(tf.image.random_flip_up_down(image))
    
    
    def random_gaussian_blur(self, image, max_sigma):
        """
        Randomly apply Gaussian blur to a TensorFlow tensor image.
        Args:
            image (Tensor): Image to be blurred.
            max_sigma (float): Maximum standard deviation of the Gaussian filter.
        Returns:
            Tensor: Blurred image.
        """
        sigma = tf.random.uniform([], 0.0, max_sigma)
        size = tf.cast(sigma * 6, tf.int32) + 1
        kernel = tf.eye(size, dtype=tf.float32)
        kernel = tf.nn.conv2d(kernel[:, :, tf.newaxis, tf.newaxis],
                            kernel[:, :, tf.newaxis, tf.newaxis],
                            strides=[1, 1, 1, 1], padding='SAME')
        kernel = kernel / tf.reduce_sum(kernel)
        return tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    
    def color_jitter(self, image, brightness=0, contrast=0, saturation=0, hue=0):
        """
        Randomly adjust the brightness, contrast, saturation, and hue of a TensorFlow tensor image.
        Args:
            image (Tensor): Image to be adjusted.
            brightness (float): Maximum brightness delta.
            contrast (float): Maximum contrast delta.
            saturation (float): Maximum saturation delta.
            hue (float): Maximum hue delta.
        Returns:
            Tensor: Adjusted image.
        """
        image = tf.image.random_brightness(image, max_delta=brightness)
        image = tf.image.random_contrast(image, lower=1-contrast, upper=1+contrast)
        image = tf.image.random_saturation(image, lower=1-saturation, upper=1+saturation)
        image = tf.image.random_hue(image, max_delta=hue)
        return tf.clip_by_value(image, 0, 255) # Clip the pixel values to [0, 255]
    
    
    def random_erasing(self, image, probability=0.5, sl=0.02, sh=0.4, r1=0.3, r2=1/0.3, max_attempts=100):
        """
        Randomly erase a rectangular region of a TensorFlow tensor image.
        Args:
            image (Tensor): Image to be erased.
            probability (float): Probability of erasing a region.
            sl (float): Minimum proportion of erased region.
            sh (float): Maximum proportion of erased region.
            r1 (float): Minimum aspect ratio of erased region.
            r2 (float): Maximum aspect ratio of erased region.
            max_attempts (int): Maximum number of attempts to generate a valid region.
        Returns:
            Tensor: Erased image.
        """
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        area = tf.cast(height * width, tf.float32)

        # Compute random parameters
        erase_prob = tf.random.uniform([], 0, 1)
        if erase_prob > probability:
            return image
        while True:
            aspect_ratio = tf.random.uniform([], r1, r2)
            h = tf.cast(tf.math.sqrt(sl * area * aspect_ratio), tf.int32)
            w = tf.cast(tf.math.sqrt(sl * area / aspect_ratio), tf.int32)
            x = tf.random.uniform([], 0, width - w, tf.int32)
            y = tf.random.uniform([], 0, height - h, tf.int32)
            if tf.cast(h * w, tf.float32) / area < sh and tf.cast(h * w, tf.float32) / area > sl:
                break
            max_attempts -= 1
            if max_attempts == 0:
                return image

        # Erase region of the image
        mask = tf.ones([h, w, tf.shape(image)[2]], dtype=tf.float32)
        erased_image = tf.tensor_scatter_nd_update(image, tf.stack([tf.range(y, y+h), tf.range(x, x+w)], axis=1), mask)
        return erased_image
    
    
    def cutout(self, image, mask_size):
        """
        Apply cutout augmentation to a TensorFlow tensor image.
        Args:
            image (Tensor): Image to be augmented.
            mask_size (int): Size of the cutout mask (width and height).
        Returns:
            Tensor: Augmented image.
        """
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        channels = tf.shape(image)[2]
        # Generate random coordinates for the cutout mask
        x = tf.random.uniform([], 0, width - mask_size, dtype=tf.int32)
        y = tf.random.uniform([], 0, height - mask_size, dtype=tf.int32)
        # Create a rectangular mask of zeros with the given size and coordinates
        mask = tf.zeros([mask_size, mask_size, channels], dtype=image.dtype)
        # Apply the mask to the image
        image = tf.tensor_scatter_nd_update(image, [[y, x, tf.constant(0)]], mask)
        return image
    
    
    def perspective_transform(self, image, max_scale):
        """
        Apply perspective transform augmentation to a TensorFlow tensor image.
        Args:
            image (Tensor): Image to be augmented.
            max_scale (float): Maximum scaling factor for perspective transform.
        Returns:
            Tensor: Augmented image.
        """
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        # Generate random coefficients for perspective transform
        x = tf.random.uniform([], -max_scale, max_scale, dtype=tf.float32)
        y = tf.random.uniform([], -max_scale, max_scale, dtype=tf.float32)
        z = tf.random.uniform([], -max_scale, max_scale, dtype=tf.float32)
        w = tf.random.uniform([], -max_scale, max_scale, dtype=tf.float32)
        # Compute the transform matrix from the coefficients
        transform = tf.stack([
            x + 1.0, y, w * width,
            z, x + 1.0, w * height,
            0.0, 0.0, 1.0,
        ], axis=0)
        transform = tf.reshape(transform, [3, 3])
        # Apply the transform to the image
        return tf.contrib.image.transform(image, transform, interpolation='BILINEAR')
    
    
    def elastic_deformation(self, image, alpha, sigma):
        """
        Apply elastic deformation augmentation to a TensorFlow tensor image.
        Args:
            image (Tensor): Image to be augmented.
            alpha (int): Scaling factor for the deformation field.
            sigma (int): Standard deviation of the Gaussian filter for smoothing the deformation field.
        Returns:
            Tensor: Augmented image.
        """
        # Generate a random deformation field
        shape = tf.shape(image)
        dx = tf.random.normal(shape, stddev=sigma) * alpha
        dy = tf.random.normal(shape, stddev=sigma) * alpha
        # Smooth the deformation field with a Gaussian filter
        filter_size = max(int(4 * sigma + 0.5), 1)
        dx = tf.keras.layers.GaussianBlur(filter_size)(dx)
        dy = tf.keras.layers.GaussianBlur(filter_size)(dy)
        # Generate the displacement grid
        x, y = tf.meshgrid(tf.range(shape[1]), tf.range(shape[0]))
        grid = tf.stack([y + dy, x + dx], axis=-1)
        # Apply the displacement to the image
        return tf.gather_nd(tf.keras.preprocessing.image.apply_affine_transform(image, grid), grid)
    
   