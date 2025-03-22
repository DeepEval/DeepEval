import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, UpSampling2D, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Downsampling

    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Stage 2: Feature Processing

    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    dropout1 = Dropout(rate=0.2)(conv4)

    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(conv5)

    # Stage 3: Upsampling and Skip Connections

    up_conv6 = UpSampling2D(size=(2, 2))(dropout2)
    up_conv6 = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(up_conv6)
    skip_conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool3)
    concat6 = Add()([up_conv6, skip_conv3])

    up_conv7 = UpSampling2D(size=(2, 2))(concat6)
    up_conv7 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(up_conv7)
    skip_conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool2)
    concat7 = Add()([up_conv7, skip_conv2])

    up_conv8 = UpSampling2D(size=(2, 2))(concat7)
    up_conv8 = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(up_conv8)
    skip_conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool1)
    concat8 = Add()([up_conv8, skip_conv1])

    # Output Layer

    conv9 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(concat8)

    model = keras.Model(inputs=input_layer, outputs=conv9)

    return model