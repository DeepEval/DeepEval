import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First branch: Dimensionality reduction using 1x1 convolution
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second branch: Feature extraction using 1x1 and 3x3 convolutions
    conv2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Third branch: Capturing larger spatial information using 1x1 and 5x5 convolutions
    conv4 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv4)

    # Fourth branch: Downsampling using 3x3 max pooling followed by 1x1 convolution
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv6 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)

    # Concatenate outputs of all branches
    concat = Concatenate()([conv1, conv2, conv3, conv4, conv5, conv6])

    # Batch normalization and flatten
    bath_norm = BatchNormalization()(concat)
    flatten = Flatten()(bath_norm)

    # Two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model