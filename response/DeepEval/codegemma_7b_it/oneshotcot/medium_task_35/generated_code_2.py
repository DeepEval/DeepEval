import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, concatenate, UpSampling2D, Conv2DTranspose, Concatenate

def dl_model():
    input_image = Input(shape=(32, 32, 3))

    # Stage 1
    conv1_1 = Conv2D(32, (3, 3), padding='same')(input_image)
    actv1_1 = Activation('relu')(conv1_1)
    conv1_2 = Conv2D(32, (3, 3), padding='same')(actv1_1)
    actv1_2 = Activation('relu')(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(actv1_2)

    # Stage 2
    conv2_1 = Conv2D(64, (3, 3), padding='same')(pool1)
    actv2_1 = Activation('relu')(conv2_1)
    conv2_2 = Conv2D(64, (3, 3), padding='same')(actv2_1)
    actv2_2 = Activation('relu')(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(actv2_2)

    # Stage 3
    conv3_1 = Conv2D(128, (3, 3), padding='same')(pool2)
    actv3_1 = Activation('relu')(conv3_1)
    conv3_2 = Conv2D(128, (3, 3), padding='same')(actv3_1)
    actv3_2 = Activation('relu')(conv3_2)
    conv3_3 = Conv2D(128, (3, 3), padding='same')(actv3_2)
    actv3_3 = Activation('relu')(conv3_3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(actv3_3)

    # Stage 4
    conv4_1 = Conv2D(256, (3, 3), padding='same')(pool3)
    actv4_1 = Activation('relu')(conv4_1)
    conv4_2 = Conv2D(256, (3, 3), padding='same')(actv4_1)
    actv4_2 = Activation('relu')(conv4_2)
    conv4_3 = Conv2D(256, (3, 3), padding='same')(actv4_2)
    actv4_3 = Activation('relu')(conv4_3)
    drop4 = Dropout(0.5)(actv4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Stage 5
    conv5_1 = Conv2D(512, (3, 3), padding='same')(pool4)
    actv5_1 = Activation('relu')(conv5_1)
    conv5_2 = Conv2D(512, (3, 3), padding='same')(actv5_1)
    actv5_2 = Activation('relu')(conv5_2)
    conv5_3 = Conv2D(512, (3, 3), padding='same')(actv5_2)
    actv5_3 = Activation('relu')(conv5_3)
    drop5 = Dropout(0.5)(actv5_3)
    pool5 = MaxPooling2D(pool_size=(2, 2))(drop5)

    # Stage 6
    up_conv6 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(pool5)
    up_conv6 = BatchNormalization()(up_conv6)
    up_conv6 = Activation('relu')(up_conv6)
    up_conv6 = Concatenate()([up_conv6, drop4])

    # Stage 7
    up_conv7 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(up_conv6)
    up_conv7 = BatchNormalization()(up_conv7)
    up_conv7 = Activation('relu')(up_conv7)
    up_conv7 = Concatenate()([up_conv7, actv3_3])

    # Stage 8
    up_conv8 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(up_conv7)
    up_conv8 = BatchNormalization()(up_conv8)
    up_conv8 = Activation('relu')(up_conv8)
    up_conv8 = Concatenate()([up_conv8, actv2_2])

    # Stage 9
    up_conv9 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(up_conv8)
    up_conv9 = BatchNormalization()(up_conv9)
    up_conv9 = Activation('relu')(up_conv9)
    up_conv9 = Concatenate()([up_conv9, actv1_2])

    # Stage 10
    conv10 = Conv2D(10, (1, 1), padding='same')(up_conv9)
    actv10 = Activation('softmax')(conv10)

    model = keras.Model(inputs=input_image, outputs=actv10)

    return model