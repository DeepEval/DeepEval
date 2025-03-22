import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense


def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first sequential block
    conv_block = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_block = BatchNormalization()(conv_block)
    conv_block = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block)
    conv_block = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_block)

    # Define the second sequential block
    conv_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block)
    conv_block2 = BatchNormalization()(conv_block2)
    conv_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block2)
    conv_block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_block2)

    # Define the third sequential block
    conv_block3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block2)
    conv_block3 = BatchNormalization()(conv_block3)
    conv_block3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block3)
    conv_block3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_block3)

    # Define the parallel branch of convolutional layers
    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch)
    conv_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_branch)

    # Define the fully connected layers
    fc_layer1 = Dense(units=512, activation='relu')(conv_block)
    fc_layer2 = Dense(units=10, activation='softmax')(fc_layer1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=fc_layer2)

    return model