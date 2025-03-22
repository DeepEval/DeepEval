# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the dl_model function
def dl_model():
    # Define the input layer with shape (32, 32, 3) for CIFAR-10 dataset
    inputs = keras.Input(shape=(32, 32, 3))

    # Define Block 1
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    x1 = layers.Conv2D(16, (1, 1), activation='relu')(x[0])
    x1 = layers.Conv2D(16, (3, 3), activation='relu')(x1)
    x1 = layers.Conv2D(16, (1, 1), activation='relu')(x1)
    
    x2 = layers.Conv2D(16, (1, 1), activation='relu')(x[1])
    x2 = layers.Conv2D(16, (3, 3), activation='relu')(x2)
    x2 = layers.Conv2D(16, (1, 1), activation='relu')(x2)
    
    x3 = layers.Conv2D(16, (1, 1), activation='relu')(x[2])
    x3 = layers.Conv2D(16, (3, 3), activation='relu')(x3)
    x3 = layers.Conv2D(16, (1, 1), activation='relu')(x3)
    
    x = layers.Concatenate(axis=-1)([x1, x2, x3])

    # Define Transition Convolution layer
    x = layers.Conv2D(16, (1, 1), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)

    # Define Block 2
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Generate channel-matching weights through two fully connected layers
    weights = layers.Dense(16, activation='relu')(x)
    weights = layers.Dense(16, activation='relu')(weights)
    
    # Reshape these weights to match the shape of adjusted output
    weights = layers.Reshape((1, 1, 16))(weights)
    
    # Multiply them with the adjusted output to produce the main path output
    x = layers.Multiply()([x, weights])

    # Connect a branch of the model directly to the input
    branch = layers.Conv2D(16, (1, 1), activation='relu')(x)
    branch = layers.AveragePooling2D((2, 2))(branch)

    # Add outputs from both the main path and the branch
    x = layers.Add()([x, branch])

    # Define the final output layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Use the dl_model function to get the constructed model
model = dl_model()
model.summary()