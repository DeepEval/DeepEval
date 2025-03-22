import keras
from keras.layers import Input, Conv2D, AvgPool2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Two blocks of convolution followed by average pooling
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = AvgPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = AvgPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2)

    # Path 2: Single convolutional layer
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs from both pathways
    path1 = Flatten()(pool2)
    path2 = Flatten()(conv3)
    combined = Add()([path1, path2])

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(combined)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model