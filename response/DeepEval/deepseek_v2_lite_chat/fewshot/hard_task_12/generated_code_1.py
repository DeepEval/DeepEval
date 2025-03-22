import keras
from keras.layers import Input, Conv2D, Add, concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 64))

    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    concat = concatenate([conv1, conv2])

    # Branch path
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Combine outputs from main and branch paths
    combined = Add()([concat, conv3])

    # Flatten the combined output
    flattened = Flatten()(combined)

    # Two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model