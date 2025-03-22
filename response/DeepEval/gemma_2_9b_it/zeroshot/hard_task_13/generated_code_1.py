import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))  

    # First Block: Feature Extraction
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    branch3 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    branch4 = layers.MaxPooling2D((3, 3), strides=(2, 2))(inputs)

    # Concatenate branches
    x = layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)

    # Second Block: Global Average Pooling and Classification
    x = layers.GlobalAveragePooling2D()(x)

    # Fully Connected Layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    # Reshape weights and element-wise multiplication
    weights = layers.Dense(32, activation='linear')(x)  
    weights = layers.Reshape((32, 32, 1))(weights) 
    x = x * weights 

    # Final Classification Layer
    outputs = layers.Dense(10, activation='softmax')(x) 

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model