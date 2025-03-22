import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: Local features
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Branch 2: Downsampling and upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: Downsampling and upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate branches
    merged = Concatenate()([branch1, branch2, branch3])

    # Final 1x1 convolution
    conv_final = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(merged)

    # Flatten and dense layers
    flatten_layer = Flatten()(conv_final)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model