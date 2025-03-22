# Import necessary libraries
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np

def dl_model():
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize pixel values
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # Define Block 1
    def block1(input_layer):
        # Define three parallel paths
        path1 = layers.AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
        path2 = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
        path3 = layers.AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)

        # Flatten the outputs and concatenate them
        flat1 = layers.Flatten()(path1)
        flat2 = layers.Flatten()(path2)
        flat3 = layers.Flatten()(path3)
        concatenated = layers.Concatenate()([flat1, flat2, flat3])
        
        return concatenated

    # Define Block 2
    def block2(input_layer):
        # Define three branches for feature extraction
        branch1 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
        branch2 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
        branch2 = layers.Conv2D(32, (1, 7), activation='relu')(branch2)
        branch2 = layers.Conv2D(32, (7, 1), activation='relu')(branch2)
        branch2 = layers.Conv2D(32, (3, 3), activation='relu')(branch2)
        branch3 = layers.AveragePooling2D(pool_size=(3, 3), strides=3)(input_layer)

        # Concatenate the outputs from all branches
        concatenated = layers.Concatenate()([branch1, branch2, branch3])
        
        return concatenated

    # Define the input layer
    input_layer = keras.Input(shape=(28, 28, 1))

    # Apply Block 1
    x = block1(input_layer)
    x = layers.Dense(128, activation='relu')(x)  # Apply a fully connected layer
    x = layers.Reshape((4, 4, 32))(x)  # Reshape the output to 4D tensor

    # Apply Block 2
    x = block2(x)

    # Apply two fully connected layers for classification
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=x)

    return model

model = dl_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])