import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 Convolutional Layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # 3x3 Depthwise Separable Convolutional Layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(conv1)

    # 1x1 Convolutional Layer to reduce dimensionality with stride of 2
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(conv2)

    # Batch Normalization
    batch_norm = BatchNormalization()(conv3)

    # Flatten the result
    flatten_layer = Flatten()(batch_norm)

    # Fully Connected Layer
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=dense)

    return model