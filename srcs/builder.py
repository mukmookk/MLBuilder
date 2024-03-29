import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics
import model
import numpy as np
import datetime
from tqdm import tqdm
import feeder as df
import config

config = config.Config()

# Define the log directory
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Create a summary writer for TensorBoard
summary_writer = tf.summary.create_file_writer(log_dir) 

# Fetch the data and preprocess it
feeder = df.CIFAR10Feeder(
        batch_size=config.batch_size, 
        val_set_size=config.val_size, 
        buffersize=config.buffer_size, 
        epochs=config.epochs, 
        image_size=config.image_size, 
        data_dir=config.image_dir
    )

train_dataset = feeder.get_train_data()
validation_dataset = feeder.get_valid_data()

# Create an instance of the ResNet model
input_shape = (32, 32, 3)
num_classes = 10

# Define model
model = model.resnet18_cifar10(input_shape, num_classes)

# Define the loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Define the optimizer
optimizer = tf.keras.optimizers.SGD()

# Load data
validationset = feeder.get_valid_data()

# Define the progress bar for the training process
progress_bar = tqdm(range(config.epochs), desc='Training', position=0)

# Compile the model with an optimizer, a loss function, and evaluation metrics
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
              loss=losses.SparseCategoricalCrossentropy(),
              metrics=[metrics.SparseCategoricalAccuracy()])

# Train the model using the fit() method 
print("Training the model with GradientTape")
for epoch in range(config.epochs):
    data_no = 0
    epoch_loss = tf.keras.metrics.Mean()
    print("Epoch {}/{}".format(epoch + 1, config.epochs))
    for X_batch, y_batch in train_dataset:
        print("No {} dataset, X_batch.shape: {}".format(data_no, X_batch.shape))
        data_no = data_no + 1
        with tf.GradientTape() as tape:
            # Compute the forward pass of the model
            logits = model(X_batch, training=True)
            # Compute the loss value for this minibatch
            loss = loss_fn(y_batch, logits)
        # Compute the gradient of the loss with respect to the weights
        gradients = tape.gradient(loss, model.trainable_weights)
        # Update the weights of the model
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # Update the loss metric with the current loss value
        epoch_loss.update_state(loss)
        
        # Update the progress bar with the current loss value
        progress_bar.set_postfix({'loss': epoch_loss.result().numpy()})
        progress_bar.update(1)

    # Reset the loss metric for the next epoch
    epoch_loss.reset_states()
    # Log the loss and accuracy values at the end of each epoch using Tensorboard
    # with summary_writer.as_default():
    #     tf.summary.scalar("loss", loss, step=optimizer.iterations)
    #     tf.summary.scalar("accuracy", metrics.SparseCategoricalAccuracy(), step=epoch)

    # Reset the loss metric for the next epoch
    epoch_loss.reset_states()

# Evaluate the model on the validation dataset
validation_loss, validation_accuracy = model.evaluate(validation_dataset)
# with summary_writer.as_default():
    # tf.summary.scalar("validation_loss", validation_loss, step=optimizer.iterations)
    # tf.summary.scalar("validation_accuracy", validation_accuracy, step=epoch)

# Save the trained model
# summary_writer.close()

