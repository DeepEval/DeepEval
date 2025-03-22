import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    drop1 = Dropout(rate=0.2)(pool1)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    drop2 = Dropout(rate=0.2)(pool2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    drop3 = Dropout(rate=0.2)(pool3)

    # Branch 4: Average pooling followed by 1x1 convolution
    pool4 = MaxPooling2D(pool_size=(8, 8), strides=8, padding='same')(input_layer)
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool4)
    drop4 = Dropout(rate=0.2)(conv4)

    # Concatenate outputs from all branches
    concatenated = Concatenate()([drop1, drop2, drop3, drop4])

    # Batch normalization and flatten
    batch_norm = BatchNormalization()(concatenated)
    flattened = Flatten()(batch_norm)

    # Three fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model