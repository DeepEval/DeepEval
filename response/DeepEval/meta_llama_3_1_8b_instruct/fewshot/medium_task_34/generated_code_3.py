import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate, Conv2DTranspose

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Part 1: Feature extraction through convolutional and max-pooling layers
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Part 2: Enhancement of generalization capabilities through convolutional, Dropout, and convolutional layers
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    drop = Dropout(0.2)(conv4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop)

    # Part 3: Upsampling through transposed convolutional layers with skip connections
    up1 = Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv5)
    skip1 = Concatenate()([up1, conv3])
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip1)
    drop1 = Dropout(0.2)(conv6)
    up2 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(drop1)
    skip2 = Concatenate()([up2, conv2])
    conv7 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip2)
    drop2 = Dropout(0.2)(conv7)
    up3 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(drop2)
    skip3 = Concatenate()([up3, conv1])
    conv8 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip3)

    # Final 1x1 convolutional layer for probability output
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv8)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model