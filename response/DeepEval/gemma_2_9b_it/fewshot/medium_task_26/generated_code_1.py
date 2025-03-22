import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 64))

    # Compress channels with 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Expand features with parallel convolutions
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv1)
    conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    concat_layer = Concatenate()([conv2_1, conv2_2])

    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model