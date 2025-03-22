import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_shape = (32, 32, 3) 
    num_classes = 10  # CIFAR-10 has 10 classes

    model = keras.Sequential()
    model.add(layers.Lambda(lambda x: tf.split(x, 3, axis=2), input_shape=input_shape))

    # Three 1x1 convolutions
    for i in range(3):
        model.add(layers.Conv2D(input_shape[2] // 3, (1, 1), activation='relu'))

    # Average pooling
    model.add(layers.AveragePooling2D((8, 8)))  

    # Concatenate the three feature maps
    model.add(layers.Lambda(lambda x: tf.concat(x, axis=2)))

    # Flatten and fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model