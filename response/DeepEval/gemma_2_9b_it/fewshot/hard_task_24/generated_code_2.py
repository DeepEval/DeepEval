import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: Local Features
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)

    # Branch 2: Downsampling & Upsampling
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool2_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2_1)
    upsample2_1 = UpSampling2D(size=(2, 2))(conv2_2)

    # Branch 3: Downsampling & Upsampling
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool3_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3_1)
    conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3_1)
    upsample3_1 = UpSampling2D(size=(2, 2))(conv3_2)

    # Concatenate Branches
    merged_features = Concatenate()([branch1, upsample2_1, upsample3_1])

    # Final 1x1 Convolution
    conv_final = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(merged_features)

    # Flatten and Fully Connected Layers
    flatten = Flatten()(conv_final)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model