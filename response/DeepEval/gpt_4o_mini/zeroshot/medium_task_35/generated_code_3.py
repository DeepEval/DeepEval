import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Stage 1: Convolution + Max Pooling
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Stage 2: Convolution + Max Pooling
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Additional Convolution Layers
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.5)(x)
    
    # Upsampling and Skip Connections
    x_skip1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x_skip1 = layers.UpSampling2D((2, 2))(x_skip1)
    
    x_skip2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x_skip1)
    x_skip2 = layers.UpSampling2D((2, 2))(x_skip2)

    # Skip Connection to the first convolution block
    x_skip2 = layers.Concatenate()([x_skip2, layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)])
    
    # Final 1x1 Convolution Layer for classification
    outputs = layers.Conv2D(10, (1, 1), activation='softmax')(x_skip2)

    # Model Definition
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()