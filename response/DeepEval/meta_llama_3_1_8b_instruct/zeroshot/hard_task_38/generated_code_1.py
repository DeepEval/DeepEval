# Import necessary packages from Keras and other libraries
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def dl_model():
    """
    This function constructs a deep learning model using Functional APIs of Keras 
    for image classification on the MNIST dataset with two processing pathways.

    Returns:
        A compiled Keras model.
    """

    # Define the input shape for the MNIST dataset (28x28 grayscale images)
    input_shape = (28, 28, 1)

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Define the first pathway
    pathway1 = layers.BatchNormalization()(inputs)
    pathway1 = layers.ReLU()(pathway1)
    pathway1 = layers.Conv2D(32, (3, 3), padding='same')(pathway1)
    pathway1 = layers.BatchNormalization()(pathway1)
    pathway1 = layers.ReLU()(pathway1)

    # Repeat the block three times
    for _ in range(2):
        pathway1 = layers.Conv2D(32, (3, 3), padding='same')(pathway1)
        pathway1 = layers.BatchNormalization()(pathway1)
        pathway1 = layers.ReLU()(pathway1)

    # Define the second pathway
    pathway2 = layers.BatchNormalization()(inputs)
    pathway2 = layers.ReLU()(pathway2)
    pathway2 = layers.Conv2D(32, (3, 3), padding='same')(pathway2)
    pathway2 = layers.BatchNormalization()(pathway2)
    pathway2 = layers.ReLU()(pathway2)

    # Repeat the block three times
    for _ in range(2):
        pathway2 = layers.Conv2D(32, (3, 3), padding='same')(pathway2)
        pathway2 = layers.BatchNormalization()(pathway2)
        pathway2 = layers.ReLU()(pathway2)

    # Concatenate the outputs from both pathways
    concatenated = layers.Concatenate()([pathway1, pathway2])

    # Apply global average pooling to reduce spatial dimensions
    x = layers.GlobalAveragePooling2D()(concatenated)

    # Define the output layer with two fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=x)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model