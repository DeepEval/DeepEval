from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Average, Multiply
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
input_shape = (32, 32, 3)


def dl_model():
    # First block layers
    conv1 = Conv2D(32, (1, 1), activation='relu', padding='same')(Input(shape=input_shape))
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv2)
    pool = MaxPooling2D((3, 3), padding='same')(conv3)

    # Concatenate outputs of parallel branches
    concat = Concatenate(axis=-1)([conv1, conv2, conv3, pool])

    # Second block layers
    flat = Flatten()(concat)
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Model building
    model = Model(inputs=[conv1, conv2, conv3, pool], outputs=[dense2])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model