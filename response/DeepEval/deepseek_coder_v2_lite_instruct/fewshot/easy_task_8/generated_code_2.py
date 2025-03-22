import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.layers import DepthwiseConv2D, SeparableConv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Depthwise Separable Convolution Layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    separable_conv = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    dropout1 = Dropout(0.25)(separable_conv)

    # 1x1 Convolution Layer
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.25)(conv1x1)

    # Max Pooling Layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(dropout2)

    # Flatten Layer
    flatten_layer = Flatten()(max_pooling)

    # Fully Connected Layer
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model