import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))
    
    # Main path
    x = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    
    # Branch 1
    branch1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    
    # Branch 2
    branch2 = layers.MaxPooling2D(pool_size=(2, 2))(x)
    branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = layers.UpSampling2D(size=(2, 2))(branch2)
    
    # Branch 3
    branch3 = layers.MaxPooling2D(pool_size=(2, 2))(x)
    branch3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = layers.UpSampling2D(size=(2, 2))(branch3)

    # Concatenate all branches
    concatenated = layers.concatenate([branch1, branch2, branch3])
    
    # Final convolution layer in main path
    output_main = layers.Conv2D(32, (1, 1), activation='relu')(concatenated)
    
    # Branch path
    branch_path = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    
    # Combine main path and branch path
    combined = layers.add([output_main, branch_path])
    
    # Flatten and add fully connected layers
    flat = layers.Flatten()(combined)
    dense1 = layers.Dense(128, activation='relu')(flat)
    output = layers.Dense(10, activation='softmax')(dense1)
    
    # Create the model
    model = models.Model(inputs=input_layer, outputs=output)
    
    return model

# Example usage:
model = dl_model()
model.summary()