import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv1 = Conv2D(32, (1, 1), padding='same', activation='sigmoid')(input_layer)
    depthwise = Conv2D(32, (3, 3), padding='same', strides=(1, 1), groups=32, activation='relu')(conv1)
    pointwise = Conv2D(64, (1, 1), padding='same', activation='relu')(depthwise)

    # Block 2
    split = Lambda(lambda x: tf.split(x, [16, 16, 16, 16], axis=-1))(input_layer)
    split_1 = Lambda(lambda x: tf.split(x, [4, 4, 4, 4], axis=-1))(split)

    # Pass through split_1 without modification
    no_modification = split_1[0]

    # Operations on split_1
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(split_1[1])
    conv2 = Conv2D(64, (5, 5), padding='same', activation='relu')(split_1[2])
    pool = MaxPooling2D(pool_size=(1, 1), padding='same')(split_1[3])

    # Concatenate outputs
    concat = Concatenate(axis=-1)([pointwise, conv1, conv2, pool])

    # Batch normalization and flattening
    batchnorm = BatchNormalization()(concat)
    flatten = Flatten()(batchnorm)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)

    return model