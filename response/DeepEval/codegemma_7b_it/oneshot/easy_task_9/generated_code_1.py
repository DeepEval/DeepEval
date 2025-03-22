import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Increase dimensionality with a 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Feature extraction with 3x3 depthwise separable convolution
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal', depthwise_constraint=keras.initializers.l2(0.01))(conv1)

    # Reduce dimensionality with another 1x1 convolution
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Add the original input to the processed output
    added = Add()([input_layer, conv3])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(added)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model