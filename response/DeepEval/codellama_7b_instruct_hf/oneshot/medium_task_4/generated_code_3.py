import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Two blocks of convolution followed by average pooling
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Path 2: Single convolutional layer
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Flatten and concatenate outputs from both paths
    flat1 = Flatten()(maxpool1)
    flat2 = Flatten()(maxpool2)
    flat3 = Flatten()(conv3)
    concatenated = Concatenate()([flat1, flat2, flat3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model