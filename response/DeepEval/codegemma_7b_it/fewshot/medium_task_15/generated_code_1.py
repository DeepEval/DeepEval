import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Extract initial features
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn = BatchNormalization()(conv)
    relu = ReLU()(bn)

    # Compress feature maps
    gap = GlobalAveragePooling2D()(relu)
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Reshape and generate weighted feature maps
    dense2_reshaped = Reshape(target_shape=(1, 1, 32))(dense2)
    multiply = Multiply()([dense2_reshaped, conv])
    concat = Concatenate()([multiply, input_layer])

    # Reduce dimensionality and downsample
    conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat)
    gap2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    gap3 = GlobalAveragePooling2D()(gap2)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(gap3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model