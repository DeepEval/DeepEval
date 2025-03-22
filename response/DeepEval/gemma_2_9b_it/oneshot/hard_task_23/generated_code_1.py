import keras
from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolution
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: Local Feature Extraction
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Branch 2 & 3: Downsampling, Convolution, Upsampling
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    up_sampling2 = UpSampling2D(size=(2, 2))(conv4)

    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    up_sampling3 = UpSampling2D(size=(2, 2))(conv5)

    # Concatenate branch outputs
    concat_layer = Concatenate()([conv3, up_sampling2, up_sampling3])

    # Final 1x1 convolution
    conv6 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(conv6)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model