import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Feature extraction through convolutional and max-pooling layers
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pool_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_1)
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pool_1)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_2)
    conv_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pool_2)
    max_pool_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_3)

    # Feature enhancement through convolutional, dropout, and convolutional layers
    conv_4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pool_3)
    drop_4 = Dropout(rate=0.4)(conv_4)
    conv_5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(drop_4)

    # Upsampling through convolutional, transposed convolutional, and skip connections
    up_6 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv_5)
    up_6 = concatenate([up_6, conv_3])
    conv_6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(up_6)
    up_7 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv_6)
    up_7 = concatenate([up_7, conv_2])
    conv_7 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(up_7)
    up_8 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv_7)
    up_8 = concatenate([up_8, conv_1])
    conv_8 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(up_8)

    # Final 1x1 convolutional layer for probability output
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='softmax')(conv_8)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model