import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Downsampling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_bn = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1_bn)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv2_bn = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2_bn)

    # Stage 2: Feature Extraction
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    conv3_bn = BatchNormalization()(conv3)
    conv3_drop = Dropout(0.25)(conv3_bn)

    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_drop)
    conv4_bn = BatchNormalization()(conv4)

    # Upsampling and Skip Connections
    up5 = UpSampling2D(size=(2, 2))(conv4_bn)
    concat5 = Concatenate()([up5, conv2_bn])
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat5)
    conv5_bn = BatchNormalization()(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5_bn)
    concat6 = Concatenate()([up6, conv1_bn])
    conv6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat6)
    conv6_bn = BatchNormalization()(conv6)

    # Output Layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv6_bn)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model