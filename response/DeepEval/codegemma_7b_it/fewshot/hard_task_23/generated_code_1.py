import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, Conv2DTranspose

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv_initial = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Branch 1: Local Feature Extraction
    conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_initial)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)

    # Branch 2: Downsampling and Upsampling
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_initial)
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool2)
    conv2_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid')(conv2_1)

    # Branch 3: Downsampling and Upsampling
    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv_initial)
    conv3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool3)
    conv3_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='valid')(conv3_1)

    # Concatenate and Refine
    concat = Concatenate()([conv1_2, conv2_2, conv3_2])
    conv_final = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Classification Layer
    flatten = Flatten()(conv_final)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model