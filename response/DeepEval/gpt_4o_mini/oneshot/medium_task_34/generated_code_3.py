import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Feature extraction part
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool2)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Processing with dropout and additional convolution
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool3)
    drop1 = Dropout(0.5)(conv4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop1)

    # Upsampling part with skip connections
    upsample1 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='valid')(conv5)
    skip1 = Concatenate()([upsample1, conv3])

    upsample2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='valid')(skip1)
    skip2 = Concatenate()([upsample2, conv2])

    upsample3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='valid')(skip2)
    skip3 = Concatenate()([upsample3, conv1])

    # Final output layer
    output_conv = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(skip3)

    model = keras.Model(inputs=input_layer, outputs=output_conv)

    return model