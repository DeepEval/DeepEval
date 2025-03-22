import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32,32,64))

    # Compress input channels with 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Expand features through parallel convolutional layers
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Concatenate results of parallel convolutional layers
    concat = Concatenate()([conv2_1, conv2_2])

    # Flatten output feature map
    flatten = Flatten()(concat)

    # Fully connected layers
    dense1 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model