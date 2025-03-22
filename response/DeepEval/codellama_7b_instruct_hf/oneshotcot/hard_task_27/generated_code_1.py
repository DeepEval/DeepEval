import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)

    # Depthwise separable convolutional layer
    depthwise_separable_conv = DepthwiseSeparableConv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv)

    # Batch normalization layer
    batch_norm = BatchNormalization()(depthwise_separable_conv)

    # Flatten layer
    flatten = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model