import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Feature extraction part
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x1 = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Save output for skip connection
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x1)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Save output for skip connection
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x2)
    x3 = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Save output for skip connection
    
    # Generalization part
    # Convolutional layer, Dropout, Convolutional layer
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x3)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)

    # Upsampling part with skip connections
    # Block 1
    x = layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.concatenate([x, x3])  # Use the saved output from Block 3

    # Block 2
    x = layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.concatenate([x, x2])  # Use the saved output from Block 2

    # Block 3
    x = layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.concatenate([x, x1])  # Use the saved output from Block 1

    # Output layer with 1x1 convolution
    outputs = layers.Conv2D(10, (1, 1), activation='softmax')(x)

    # Model construction
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# You can now create the model
model = dl_model()
model.summary()