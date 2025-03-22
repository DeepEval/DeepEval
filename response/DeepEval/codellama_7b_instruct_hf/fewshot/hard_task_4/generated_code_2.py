import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, GlobalAveragePooling2D, Dense, Flatten

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 convolution to increase dimensionality
    conv1 = Conv2D(filters=3 * 3, kernel_size=1, strides=1, padding='same')(input_layer)

    # Depthwise separable convolution to extract initial features
    conv2 = DepthwiseSeparableConv2D(filters=3 * 3, kernel_size=3, strides=1, padding='same')(conv1)

    # Global average pooling to compute channel attention weights
    pool1 = GlobalAveragePooling2D(pool_size=2, strides=2)(conv2)

    # Fully connected layers to generate channel attention weights
    dense1 = Dense(units=3 * 3, activation='relu')(pool1)
    dense2 = Dense(units=3 * 3, activation='relu')(dense1)

    # Reshape channel attention weights to match initial features
    reshape = Reshape(target_shape=(3, 3))(dense2)

    # Multiply channel attention weights with initial features
    conv3 = Conv2D(filters=3 * 3, kernel_size=1, strides=1, padding='same')(reshape)

    # 1x1 convolution to reduce dimensionality
    conv4 = Conv2D(filters=3 * 3, kernel_size=1, strides=1, padding='same')(conv3)

    # Flatten layer to combine output with input
    flatten = Flatten()(conv4)

    # Fully connected layer for classification
    dense3 = Dense(units=10, activation='softmax')(flatten)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=dense3)

    return model