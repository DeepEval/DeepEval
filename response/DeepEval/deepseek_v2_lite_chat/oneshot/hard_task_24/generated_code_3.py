import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are of size 32x32
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: 3x3 convolutional layer
    conv_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: Max Pooling -> 3x3 conv -> UpSampling
    max_pool_branch2 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(input_layer)
    conv_branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool_branch2)
    up_branch2 = UpSampling2D(size=(2, 2))(conv_branch2)

    # Branch 3: Max Pooling -> 5x5 conv -> UpSampling
    max_pool_branch3 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(input_layer)
    conv_branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(max_pool_branch3)
    up_branch3 = UpSampling2D(size=(2, 2))(conv_branch3)

    # Concatenate all branches
    fuse_branch1 = Concatenate()([conv_branch1, conv_branch2, conv_branch3])

    # Final 1x1 convolutional layer
    conv_final = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fuse_branch1)

    # Output fully connected layers
    flatten_layer = Flatten()(conv_final)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model