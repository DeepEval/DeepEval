# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model includes three convolutional layers, where the output from each layer is concatenated with the input of the previous layer along the channel dimension for the next layer.
    Finally, the output will be flattened and passed through two fully connected layers to produce classification probabilities, making the model suitable for multi-class classification tasks.
    
    Parameters:
    None
    
    Returns:
    model (Model): The constructed deep learning model.
    """

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Define the input shape
    input_shape = x_train.shape[1:]

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Convolutional layer 1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)

    # Convolutional layer 2
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

    # Convolutional layer 3
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    # Concatenate the outputs of the convolutional layers
    concatenated = Concatenate()([conv1, conv2, conv3])

    # Flatten the output
    flattened = Flatten()(concatenated)

    # Dense layer 1
    dense1 = Dense(128, activation='relu')(flattened)

    # Dense layer 2 (output layer)
    outputs = Dense(10, activation='softmax')(dense1)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model