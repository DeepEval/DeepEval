import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)
    inputs = layers.Input(shape=input_shape)
    
    # Part 1: Convolutional layers with max pooling
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Part 2: Additional Convolutional layers with Dropout
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Part 3: Upsampling with transposed convolutions and skip connections
    skip_connection_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, skip_connection_1])
    
    skip_connection_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(skip_connection_1)
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, skip_connection_2])
    
    skip_connection_3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(skip_connection_2)
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, skip_connection_3])
    
    # Final 1x1 convolutional layer for classification
    x = layers.Conv2D(10, (1, 1), activation='softmax')(x)
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=x)
    
    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()  # Display the model summary