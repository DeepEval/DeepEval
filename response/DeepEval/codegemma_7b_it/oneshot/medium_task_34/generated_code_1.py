import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Feature Extraction

    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Feature Enhancement

    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    drop4 = Dropout(0.5)(conv3)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop4)

    # Upsampling

    upsample5 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='valid')(conv4)
    skip_connection5 = Concatenate()([upsample5, conv2])
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip_connection5)

    upsample6 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid')(conv5)
    skip_connection6 = Concatenate()([upsample6, conv1])
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip_connection6)

    # Output Layer

    conv7 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='valid')(conv6)
    output_layer = Flatten()(conv7)

    # Model Definition

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model