import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(act1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)

    # Block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(act2)
    bn3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(bn3)

    # Concatenate outputs of blocks
    concat_layer = Concatenate(axis=3)([act1, act2, act3])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model