import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D
from keras.layers import Reshape, Lambda

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)  # initial convolutional layer

    # Branch 1: extracts local features
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)

    # Branch 2: downsamples and then upsamples
    downsampling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)
    conv_branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(downsampling)
    upsampling = UpSampling2D(size=(2, 2))(conv_branch2)

    # Branch 3: downsamples and then upsamples
    downsampling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)
    conv_branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(downsampling)
    upsampling = UpSampling2D(size=(2, 2))(conv_branch3)

    # Fuse the outputs of all branches
    output_tensor = Concatenate()([branch1, upsampling, upsampling])

    # Apply another convolutional layer
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)

    # Flatten the output
    flatten_layer = Flatten()(conv)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model