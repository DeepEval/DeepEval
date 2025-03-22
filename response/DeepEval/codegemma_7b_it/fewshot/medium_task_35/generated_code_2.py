import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, UpSampling2D, Dropout, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Stage 1
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)

    # Stage 2
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_2)

    # Additional Convolutions and Dropout
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    drop1 = Dropout(rate=0.2)(conv3)

    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop1)
    drop2 = Dropout(rate=0.2)(conv4)

    # Skip Connections
    up1 = UpSampling2D(size=(2, 2))(drop2)
    up1 = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(up1)
    skip1 = Add()([up1, conv2_2])

    up2 = UpSampling2D(size=(2, 2))(skip1)
    up2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(up2)
    skip2 = Add()([up2, conv1_2])

    # Final Layer
    conv5 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(skip2)
    output_layer = Flatten()(conv5)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model