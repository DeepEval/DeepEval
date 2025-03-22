import keras
from keras import backend as K
from keras.layers import Input, Conv2D, BatchNormalization, Lambda, Add, concatenate, Flatten, Dense

def depthwise_separable_conv(input_tensor):
    # Elevation
    elevation = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    # Depthwise convolution
    depthwise = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(elevation)

    # Pointwise convolution
    pointwise = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise)

    return pointwise

def block(input_tensor):
    # Depthwise separable convolution
    conv = depthwise_separable_conv(input_tensor)

    # Identity mapping
    identity = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    # Addition
    output_tensor = Add()([conv, identity])

    return output_tensor

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    branch1 = block(input_layer)
    branch1 = BatchNormalization()(branch1)

    # Branch 2
    branch2 = block(input_layer)
    branch2 = BatchNormalization()(branch2)

    # Branch 3
    branch3 = block(input_layer)
    branch3 = BatchNormalization()(branch3)

    # Concatenate branches
    concat = concatenate([branch1, branch2, branch3])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model