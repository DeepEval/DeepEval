# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

# Define the function to create the deep learning model
def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model extracts features at multiple scales using various convolution and pooling operations,
    including 1x1, 3x3, and 5x5 convolutions, as well as 3x3 max pooling. These features are combined
    through concatenation. Finally, the model outputs classification results via a flattening layer
    followed by two fully connected layers.
    
    Returns:
        A constructed Keras model.
    """

    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create the base model using the Functional API
    inputs = keras.Input(shape=input_shape)

    # Convolution and pooling block 1
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2)(x)

    # Convolution and pooling block 2
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2)(x)

    # Convolution and pooling block 3
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2)(x)

    # Feature extraction using 1x1, 3x3, and 5x5 convolutions
    x_1x1 = layers.Conv2D(128, (1, 1), activation='relu')(x)
    x_3x3 = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x_5x5 = layers.Conv2D(128, (5, 5), activation='relu')(x)

    # Concatenate features
    x = layers.Concatenate()([x_1x1, x_3x3, x_5x5])

    # Convolution and pooling block 4
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2)(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model