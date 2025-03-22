import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()
    
    # Input Layer
    input_tensor = keras.Input(shape=(32, 32, 3))  # Assuming CIFAR-10 shape

    # Reshape and Permute for Channel Shuffling
    x = layers.Reshape((32, 32, 3, 1))(input_tensor)  
    x = layers.Permute((2, 3, 1, 4))(x)  
    x = layers.Reshape((32, 32, 3))(x) 

    # Fully Connected Layer and Output
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=x)
    return model