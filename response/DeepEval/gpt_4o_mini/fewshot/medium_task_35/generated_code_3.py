import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels (RGB)

    # Stage 1: Convolution and MaxPooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # Downsampling

    # Stage 2: Convolution and MaxPooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # Downsampling

    # Additional Convolutions and Dropout
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.5)(conv3)  # Dropout layer for regularization
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv3)

    # UpSampling with Skip Connections
    upsample1 = UpSampling2D(size=(2, 2))(conv3)
    skip1 = Concatenate()([upsample1, conv2])  # Skip connection from conv2
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(skip1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv4)

    upsample2 = UpSampling2D(size=(2, 2))(conv4)
    skip2 = Concatenate()([upsample2, conv1])  # Skip connection from conv1
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(skip2)
    
    # Final 1x1 Convolution to produce probability outputs
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv5)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model