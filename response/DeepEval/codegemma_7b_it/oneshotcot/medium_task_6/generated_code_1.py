import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels (R, G, B)

    # Initial convolution
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Three parallel blocks
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    block1 = BatchNormalization()(block1)
    block1 = keras.layers.Activation('relu')(block1)

    block2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    block2 = BatchNormalization()(block2)
    block2 = keras.layers.Activation('relu')(block2)

    block3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    block3 = BatchNormalization()(block3)
    block3 = keras.layers.Activation('relu')(block3)

    # Add outputs of blocks to initial convolution
    output = Concatenate()([conv, block1, block2, block3])

    # Flatten and fully connected layers
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=64, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model