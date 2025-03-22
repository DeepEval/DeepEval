import keras
from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, Conv2DTranspose, BatchNormalization, Flatten, Dense
from keras import backend as K

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: Local feature extraction
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Branch 2: Downsampling, upsampling
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    upsample = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Branch 3: Downsampling, upsampling
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    upsample = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    # Concatenate and refine
    output_tensor = Concatenate()([conv2, upsample, upsample])
    refine = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)

    flatten_layer = Flatten()(refine)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model