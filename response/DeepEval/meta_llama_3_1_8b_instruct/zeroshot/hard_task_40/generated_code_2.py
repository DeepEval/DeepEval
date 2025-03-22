# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Block 1: Average Pooling
    x = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Reshape the output to 4 dimensions
    x = layers.Reshape((3 * 3 * 1,))(x)

    # Block 2: Multi-scale feature extraction
    path_1 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    path_1 = layers.Dropout(0.2)(path_1)
    path_1 = layers.Flatten()(path_1)

    path_2 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    path_2 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(path_2)
    path_2 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(path_2)
    path_2 = layers.Dropout(0.2)(path_2)
    path_2 = layers.Flatten()(path_2)

    path_3 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    path_3 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(path_3)
    path_3 = layers.Dropout(0.2)(path_3)
    path_3 = layers.Flatten()(path_3)

    path_4 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    path_4 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(path_4)
    path_4 = layers.Dropout(0.2)(path_4)
    path_4 = layers.Flatten()(path_4)

    # Concatenate the outputs from all paths
    x = layers.Concatenate()([path_1, path_2, path_3, path_4])

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
print(model.summary())