# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model using the Functional API of Keras.
    The model is designed for image classification using the CIFAR-10 dataset.
    
    It consists of three convolutional layers, where the output from each layer is concatenated 
    with the input of the previous layer along the channel dimension for the next layer.
    
    Finally, the output is flattened and passed through two fully connected layers to produce 
    classification probabilities, making the model suitable for multi-class classification tasks.
    
    Returns:
        A Keras model instance.
    """

    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Define the first convolutional layer with max pooling
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Define the second convolutional layer with max pooling
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Define the third convolutional layer with max pooling
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Concatenate the output from each convolutional layer with the input of the previous layer
    x = layers.Concatenate()([x, x, x])

    # Flatten the output
    x = layers.Flatten()(x)

    # Define the first fully connected layer
    x = layers.Dense(128, activation='relu')(x)

    # Define the second fully connected layer
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Create a deep learning model
model = dl_model()
model.summary()