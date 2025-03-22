import keras
from keras.layers import Input, Conv2D, AveragePooling2D, TransposedConv2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: Local Feature Extraction
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Branch 2 & 3: Upsampling & Downsampling
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    upsample1 = TransposedConv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    upsample2 = TransposedConv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5)

    # Concatenate Branches
    concat_layer = Concatenate()([conv3, upsample1, upsample2])
    final_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)

    # Flatten & Classification
    flatten_layer = Flatten()(final_conv)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model