import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Increase dimensionality with 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Feature extraction with 3x3 depthwise separable convolution
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_uniform')(conv1)

    # Reduce dimensionality with 1x1 convolution and apply stride of 2
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(conv2)

    # Flatten and output classification probabilities
    flatten_layer = Flatten()(conv3)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model