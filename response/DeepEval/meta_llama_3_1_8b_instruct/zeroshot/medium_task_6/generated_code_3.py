from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize input data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Convert class labels to categorical labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define input layer
    inputs = Input(shape=(32, 32, 3))

    # Initial convolution
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

    # Block 1
    block1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    block1 = BatchNormalization()(block1)
    block1 = Conv2D(32, (3, 3), activation='relu', padding='same')(block1)
    block1 = BatchNormalization()(block1)

    # Block 2
    block2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    block2 = BatchNormalization()(block2)
    block2 = Conv2D(64, (3, 3), activation='relu', padding='same')(block2)
    block2 = BatchNormalization()(block2)

    # Block 3
    block3 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    block3 = BatchNormalization()(block3)
    block3 = Conv2D(128, (3, 3), activation='relu', padding='same')(block3)
    block3 = BatchNormalization()(block3)

    # Add outputs of blocks
    x = Add()([x, block1, block2, block3])

    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs, x)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model