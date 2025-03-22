import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Conv2DTranspose, Concatenate, BatchNormalization, Dropout, Activation
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Downsampling
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)

    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_2)

    # Additional layers
    conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
    drop3 = Dropout(0.5)(conv3_2)

    # Stage 2: Upsampling with skip connections
    deconv2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(drop3)
    concat2 = Concatenate()([conv2_2, deconv2])
    conv4_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat2)
    conv4_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4_1)

    deconv1 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv4_2)
    concat1 = Concatenate()([conv1_2, deconv1])
    conv5_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat1)
    conv5_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5_1)

    # Output layer
    conv6 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv5_2)

    model = Model(inputs=input_layer, outputs=conv6)
    return model