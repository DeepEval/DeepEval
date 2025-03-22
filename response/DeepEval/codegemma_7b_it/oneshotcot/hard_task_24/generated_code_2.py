import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, ZeroPadding2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    initial_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Branch 1: Local feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)

    # Branch 2: Max pooling downsampling and upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(initial_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3: Max pooling downsampling and upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(initial_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(4, 4))(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Branch 4: Max pooling downsampling
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(initial_conv)
    branch4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch4)

    # Concatenate branches and final convolutional layer
    concat = Concatenate()([branch1, branch2, branch3, branch4])
    final_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(concat)

    # Fully connected layers
    flatten = Flatten()(final_conv)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model