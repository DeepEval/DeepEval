import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv1)

    # Third convolutional layer
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv2)

    # Fourth convolutional layer
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv3)

    # Fifth convolutional layer
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv4)

    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv5)

    # Concatenate the output of the max pooling layer with the output of the first convolutional layer
    concatenated = Concatenate()([max_pooling, conv1])

    # Batch normalization layer
    batch_normalization = BatchNormalization()(concatenated)

    # Flatten the output of the batch normalization layer
    flattened = Flatten()(batch_normalization)

    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(flattened)

    # Second fully connected layer
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model