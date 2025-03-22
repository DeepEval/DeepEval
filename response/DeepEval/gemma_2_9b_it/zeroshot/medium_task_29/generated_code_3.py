import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.Sequential()

    # Input layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    
    # 1x1 Max Pooling
    model.add(layers.MaxPooling2D((1, 1), strides=(1, 1)))
    
    # Flatten the output
    model.add(layers.Flatten())

    # 2x2 Max Pooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Flatten())

    # 4x4 Max Pooling
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((4, 4), strides=(4, 4)))
    model.add(layers.Flatten())

    # Concatenate flattened features
    model.add(layers.Concatenate())

    # Fully Connected Layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model