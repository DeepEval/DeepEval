import keras
from keras.layers import Input, Conv2D, AveragePooling2D, TransposedConv2D, Concatenate, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))
    
    # 1x1 Convolution
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: Local Feature Extraction
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)

    # Branch 2: Downsampling & Upsampling
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    upsample3 = TransposedConv2D(filters=64, kernel_size=(2, 2), strides=2, padding='same')(conv3_1)

    # Branch 3: Downsampling & Upsampling
    pool4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv5_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool4)
    upsample5 = TransposedConv2D(filters=64, kernel_size=(2, 2), strides=2, padding='same')(conv5_1)

    # Concatenate Branch Outputs
    merge = Concatenate()([conv2_2, upsample3, upsample5])
    conv6 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(merge)

    # Flatten and Fully Connected Layers
    flatten = Flatten()(conv6)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model