import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution
    conv_init = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn_init = BatchNormalization()(conv_init)
    act_init = Activation('relu')(bn_init)

    # Block 1
    conv_block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(act_init)
    bn_block1 = BatchNormalization()(conv_block1)
    act_block1 = Activation('relu')(bn_block1)

    # Block 2
    conv_block2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(act_block1)
    bn_block2 = BatchNormalization()(conv_block2)
    act_block2 = Activation('relu')(bn_block2)

    # Block 3
    conv_block3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(act_block2)
    bn_block3 = BatchNormalization()(conv_block3)
    act_block3 = Activation('relu')(bn_block3)

    # Adding outputs of blocks to initial convolution's output
    added_output = concatenate([act_init, act_block1, act_block2, act_block3])

    # Max pooling and flattening
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(added_output)
    flattened = Flatten()(max_pool)

    # Fully connected layers and classification
    dense_layer1 = Dense(units=128, activation='relu')(flattened)
    dense_layer2 = Dense(units=64, activation='relu')(dense_layer1)
    output_layer = Dense(units=10, activation='softmax')(dense_layer2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model