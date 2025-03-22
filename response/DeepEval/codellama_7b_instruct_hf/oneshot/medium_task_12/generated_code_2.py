import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    block1_output = Concatenate(axis=1)([conv1, conv2, conv3, maxpool])

    # Block 2
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv6)
    block2_output = Concatenate(axis=1)([conv4, conv5, conv6, maxpool2])

    # Block 3
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    conv8 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv7)
    conv9 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv8)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv9)
    block3_output = Concatenate(axis=1)([conv7, conv8, conv9, maxpool3])

    # Flatten and fully connected layers
    flatten = Flatten()(block3_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model