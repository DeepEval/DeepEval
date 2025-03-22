import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def dl_model():
    """
    Create a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model employs two stages of convolution and max pooling for downsampling, 
    enabling multi-level feature extraction. These features are then processed through 
    additional convolutional and dropout layers. To restore spatial information, the 
    model integrates two sets of convolutions with upsampling using skip connections 
    to the corresponding convolutional layers. Finally, it produces probability outputs 
    for 10 categories through a 1x1 convolutional layer.
    
    Returns:
        model: The constructed Keras model.
    """

    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Stage 1: Convolution and max pooling for downsampling
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Stage 2: Additional convolutional and dropout layers
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Stage 3: Convolutions with upsampling using skip connections
    x_skip1 = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x_skip1 = layers.Conv2D(128, (3, 3), activation='relu')(x_skip1)
    x_upsample1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu')(x_skip1)
    x = layers.Concatenate()([x, x_upsample1])

    x_skip2 = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x_skip2 = layers.Conv2D(64, (3, 3), activation='relu')(x_skip2)
    x_upsample2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu')(x_skip2)
    x = layers.Concatenate()([x, x_upsample2])

    # Output layer: 1x1 convolutional layer for probability outputs
    outputs = layers.Conv2D(10, (1, 1), activation='softmax')(x)
    outputs = layers.GlobalAveragePooling2D()(outputs)  # Global average pooling to get the final output

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model