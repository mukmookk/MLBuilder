import tensorflow as tf
import dataFeeder as df
from tensorflow.keras import layers, optimizers, losses, metrics
import model as m
import numpy as np
import math

epochs = 5
train_set_size = 64
# Load the data and preprocess it
feeder = df.CIFAR10Feeder(train_set_size=train_set_size, test_set_size=train_set_size, buffersize=2000, epochs=epochs, image_size=(32, 32), data_dir="./cifar-10-batches-py")
train_dataset = feeder.get_train_data()
test_dataset = feeder.get_valid_data()

# Create an instance of the ResNet model
input_shape = (32, 32, 3)
num_classes = 10
model = m.resnet18_cifar10(input_shape, num_classes)

# Compile the model with an optimizer, a loss function, and evaluation metrics
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
              loss=losses.SparseCategoricalCrossentropy(),
              metrics=[metrics.SparseCategoricalAccuracy()])

steps_per_epoch = math.ceil(50000 / feeder.train_batch_size)

# Train the model using the fit() method 
history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=test_dataset)

test_dataset = feeder.get_valid_data()

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

# Save the trained model
model.save("resnet18_cifar10.h5")