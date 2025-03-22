# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

def dl_model():
    """
    This function creates a deep learning model for image classification using the CIFAR-10 dataset.
    The model processes the input image by splitting it into three groups along the channel dimension 
    and applying different convolutional kernels for multi-scale feature extraction.
    
    Returns:
        A constructed model.
    """

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Define the input shape of the model
    input_shape = (32, 32, 3)

    # Create the model
    model = keras.Model(inputs=keras.Input(shape=input_shape), outputs=[])
    
    # Define a Lambda layer to split the input image into three groups along the channel dimension
    split_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))
    
    # Add the Lambda layer to the model
    model.add(split_layer)
    
    # Define three groups of convolutional layers with different kernel sizes
    model.add(layers.Conv2D(32, (1, 1), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Use the split_layer to split the model into three groups
    group1 = keras.Model(inputs=model.inputs, outputs=model.layers[1].output)
    group2 = keras.Model(inputs=model.inputs, outputs=model.layers[5].output)
    group3 = keras.Model(inputs=model.inputs, outputs=model.layers[9].output)
    
    # Add the groups to the model
    model.add(layers.Concatenate([group1.output, group2.output, group3.output]))
    
    # Flatten the output of the model
    model.add(layers.Flatten())
    
    # Add two fully connected layers for classification
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Usage example
model = dl_model()
model.summary()