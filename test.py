# import pytest ## https://towardsdatascience.com/pytest-for-machine-learning-a-simple-example-based-tutorial-a3df3c58cf8
from basic_feeder_using_tf import VisionFeeder
import matplotlib.pyplot as plt
import pickle
import numpy as np

# # Create a VisionFeeder object
# vision_feeder = VisionFeeder(image_dir="./images", resize_shape=(256, 256))

# # Get a tf.data.Dataset object for training
# train_dataset = vision_feeder.train_dataset(batch_size=2)

# # Get a tf.data.Dataset object for validation
# test = vision_feeder.valid_dataset(batch_size=2)


# for batch in train_dataset.take(1):
#     print(batch.shape)
    
#     image = batch[0]
#     plt.imshow(image)
#     plt.show()

def load_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == "__main__":
    train_batches = [load_batch('./cifar-10-batches-py/data_batch_{}'.format(i)) for i in range(1, 6)]
    X_train = np.concatenate([batch[b'data'] for batch in train_batches], axis=0)
    y_train = np.concatenate([batch[b'labels'] for batch in train_batches], axis=0)
    
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # show images
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(X_train[i])
        ax.set_title(class_names[y_train[i]])
        ax.axis('off')
    plt.show()