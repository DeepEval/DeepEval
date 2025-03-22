from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    model = keras.Sequential(
        [
            layers.Conv2D(32, (1, 1), input_shape=(28, 28, 1)),
            layers.Activation('relu'),
            layers.DepthwiseConv2D(3, (3, 3)),
            layers.Activation('relu'),
            layers.Conv2D(16, (1, 1), strides=2),
            layers.Activation('relu'),
            layers.Flatten(),
            layers.Dense(10, activation='softmax')
        ]
    )

    return model