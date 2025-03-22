import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, UpSampling2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has 32x32 images with 3 color channels

    # Initial 1x1 convolutional layer
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: extracts local features through a 3x3 convolutional layer
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)

    # Branch 2: downsamples through max pooling, then extracts features through a 3x3 convolutional layer, and upsamples
    downsample2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(downsample2)
    upsample2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: downsamples through max pooling, then extracts features through a 3x3 convolutional layer, and upsamples
    downsample3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(downsample3)
    upsample3 = UpSampling2D(size=(2, 2))(branch3)

    # Fuse the outputs of all branches through concatenation
    fused_output = Concatenate()([branch1, upsample2, upsample3])

    # Apply another 1x1 convolutional layer
    conv_fused = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fused_output)

    # Flatten the output
    flatten_layer = Flatten()(conv_fused)

    # Pass through three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model