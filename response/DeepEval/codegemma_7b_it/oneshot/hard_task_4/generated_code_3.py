import keras
from keras import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply, Conv2DTranspose, Activation, ZeroPadding2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Increase dimensionality of input's channels threefold with a 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Extract initial features using a 3x3 depthwise separable convolution
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False)(conv1)
    bn2 = BatchNormalization()(conv2)
    actv2 = Activation('relu')(bn2)

    # Compute channel attention weights through global average pooling followed by two fully connected layers, generating weights whose size is same as the channels of the initial features
    avg_pool = GlobalAveragePooling2D()(actv2)
    dense1 = Dense(units=32, activation='relu')(avg_pool)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)

    # Reshape weights to match the initial features and multiply with initial features to achieve channel attention weighting
    reshape = Reshape(target_shape=(32, 32, 32))(dense2)
    multiply = Multiply()([actv2, reshape])

    # 1x1 convolution reduces dimensionality
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(multiply)

    # Combine output with initial input
    concat = Concatenate()([conv1, conv3])

    # Flatten and fully connected layer for classification
    flatten = Flatten()(concat)
    dense4 = Dense(units=32, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense4)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model