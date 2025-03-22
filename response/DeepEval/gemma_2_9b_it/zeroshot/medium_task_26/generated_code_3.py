from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()
    
    # 1x1 Convolutional Layer for Channel Compression
    model.add(layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu', input_shape=(32, 32, 64))) 

    # Parallel 1x1 and 3x3 Convolutional Layers
    model.add(layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

    # Concatenate Feature Maps
    model.add(layers.concatenate([layers.Lambda(lambda x: x), layers.Lambda(lambda x: x)])) 

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dense(units=10, activation='softmax'))  # Assuming 10 classes

    return model