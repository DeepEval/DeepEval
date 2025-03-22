import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation
from keras.models import Model
from keras.datasets import cifar10
import numpy as np

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    norm1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(norm1)

    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    norm2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(norm2)

    # Block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    norm3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(norm3)

    # Concatenate the outputs of the three blocks
    concat = Concatenate()(list(pool3.output) + list(conv3.output))

    # Flatten and pass through two fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])