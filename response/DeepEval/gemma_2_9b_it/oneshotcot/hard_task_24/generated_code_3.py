import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():  

    input_layer = Input(shape=(32, 32, 3)) 

    # 1x1 initial convolution
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(input_layer)

    # Branch 1: Local features
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)

    # Branch 2: Downsampling & Upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: Downsampling & Upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2))(conv1)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate branches
    merged = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolution to fuse
    merged = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(merged)

    # Flatten and dense layers
    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model