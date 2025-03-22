import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 64))

    # Main path
    # 1x1 Convolution for dimensionality reduction
    x_main = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # Two parallel convolutional layers
    x_1x1 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x_main)
    x_3x3 = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x_main)

    # Concatenate outputs from the two convolutional layers
    x_main_concat = layers.concatenate([x_1x1, x_3x3])

    # Branch path
    x_branch = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)

    # Combine main and branch paths
    x_combined = layers.add([x_main_concat, x_branch])

    # Flatten the combined output
    x_flatten = layers.Flatten()(x_combined)

    # Fully connected layers
    x_fc1 = layers.Dense(128, activation='relu')(x_flatten)
    x_fc2 = layers.Dense(10, activation='softmax')(x_fc1)  # Assuming 10 classes for classification

    # Construct the model
    model = models.Model(inputs=inputs, outputs=x_fc2)

    return model

# Example of how to use the function to create the model
model = dl_model()
model.summary()  # To print the model summary