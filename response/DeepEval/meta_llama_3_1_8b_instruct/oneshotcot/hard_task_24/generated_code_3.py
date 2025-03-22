import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D
from keras.layers import concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer
    conv_initial = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Branch 1: 3x3 convolutional layer
    conv_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_initial)

    # Branch 2: max pooling, 3x3 convolution, and upsampling
    max_pool_branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_initial)
    conv_branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool_branch2)
    upsample_branch2 = UpSampling2D(size=(2, 2))(conv_branch2)

    # Branch 3: max pooling, 3x3 convolution, and upsampling
    max_pool_branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_initial)
    conv_branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool_branch3)
    upsample_branch3 = UpSampling2D(size=(2, 2))(conv_branch3)

    # Concatenate the outputs of all branches
    output_branches = concatenate([conv_branch1, upsample_branch2, upsample_branch3], axis=3)

    # 1x1 convolutional layer
    conv_final = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(output_branches)

    # Flatten the output
    flatten_layer = Flatten()(conv_final)

    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model