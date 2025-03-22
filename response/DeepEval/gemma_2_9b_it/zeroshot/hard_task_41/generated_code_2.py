import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))  

    # Block 1: Parallel Paths
    branch1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    branch2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    branch3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    branch1 = layers.Flatten()(branch1)
    branch2 = layers.Flatten()(branch2)
    branch3 = layers.Flatten()(branch3)

    branch1 = layers.Dropout(0.25)(branch1)  
    branch2 = layers.Dropout(0.25)(branch2) 
    branch3 = layers.Dropout(0.25)(branch3) 

    x = layers.Concatenate()([branch1, branch2, branch3]) 

    # Fully Connected Layer and Reshape
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((1, 128))(x)

    # Block 2: Branch Connections
    branch4_1 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    branch4_2 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    branch4_3 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(branch4_2)

    branch5_1 = layers.Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    branch5_2 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(branch5_1)
    branch5_3 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(branch5_2)

    branch6_1 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    branch6_2 = layers.Conv2D(64, kernel_size=(1, 1), activation='relu')(branch6_1)

    x = layers.Concatenate()([branch4_1, branch4_3, branch5_3, branch6_2])

    # Final Classification Layers
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)  

    model = keras.Model(inputs=inputs, outputs=x)
    return model