import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Add
from keras.layers import Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First stage of convolution and pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Second stage of convolution and pooling
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    # Additional convolutional and dropout layers
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    drop1 = Dropout(rate=0.5)(conv5)

    # Upsampling and skip connections
    up1 = UpSampling2D(size=(2, 2))(drop1)
    skip1 = Add()([up1, conv4])  # skip connection to conv4
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(skip1)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    skip2 = Add()([up2, conv2])  # skip connection to conv2
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(skip2)

    # Final 1x1 convolution to get the probability outputs for 10 categories
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax', padding='same')(conv7)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model