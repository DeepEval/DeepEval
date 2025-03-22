import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: Extract local features through a 3x3 convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: Downsampling, 3x3 convolution, and upsampling
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool2)
    upsample2 = UpSampling2D(size=(2, 2))(conv2)

    # Branch 3: Downsampling, 3x3 convolution, and upsampling
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool3)
    upsample3 = UpSampling2D(size=(2, 2))(conv3)

    # Fuse the outputs of all branches through concatenation
    concat = Concatenate()([conv1, upsample2, upsample3])

    # Final 1x1 convolutional layer
    final_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Flatten and pass through three fully connected layers
    flatten = Flatten()(final_conv)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model