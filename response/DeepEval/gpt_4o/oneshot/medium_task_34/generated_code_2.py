import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add, Activation

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    # Feature extraction with 3 pairs of <convolutional layer, max-pooling layer>
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Processing with <convolutional layer, Dropout layer, convolutional layer>
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    dropout = Dropout(rate=0.5)(conv4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(dropout)

    # Upsampling with 3 pairs of <convolutional layer, transposed convolutional layer>, with skip connections
    up1 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv5)
    skip1 = Add()([up1, conv3])  # Skip connection

    up2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(skip1)
    skip2 = Add()([up2, conv2])  # Skip connection

    up3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(skip2)
    skip3 = Add()([up3, conv1])  # Skip connection

    # Final 1x1 convolutional layer for classification
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax', padding='same')(skip3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model