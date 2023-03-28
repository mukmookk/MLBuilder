import tensorflow as tf
import numpy as np

def print_iterator_items(iterator, num_items=5):
    counter = 0
    for item in iterator:
        print(item)
        counter += 1
        if counter >= num_items:
            break
        
# Convert PIL images to TensorFlow tensors
def convert_pil_to_tensor(image):
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Add a batch dimension
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    return image_tensor