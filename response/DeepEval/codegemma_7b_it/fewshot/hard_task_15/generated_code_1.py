import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Multiply, Reshape, concatenate
import keras.backend as K

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # Main Path
    conv = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    gap = GlobalAveragePooling2D()(conv)
    fc = Dense(64, activation='relu')(gap)
    weights = Dense(64)(fc)
    weights = Reshape((1, 1, 64))(weights)
    weighted_conv = Multiply()([inputs, weights])

    # Branch Path
    branch = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)

    # Merging
    combined = concatenate([weighted_conv, branch])

    # Fully Connected Layers
    fc1 = Dense(64, activation='relu')(combined)
    outputs = Dense(10, activation='softmax')(fc1)

    model = Model(inputs, outputs)
    return model