import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Part 1: Feature Extraction
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_1)
    
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)
    
    conv1_3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_3)

    # Part 2: Enhance Generalization
    conv2_1 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    dropout = Dropout(rate=0.5)(conv2_1)
    conv2_2 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(dropout)

    # Part 3: Upsampling with Skip Connections
    upconv1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2_2)
    upsample1 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same')(upconv1)
    skip1 = Add()([upsample1, conv1_3])  # Skip connection

    upconv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(skip1)
    upsample2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(upconv2)
    skip2 = Add()([upsample2, conv1_2])  # Skip connection

    upconv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(skip2)
    upsample3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(upconv3)
    skip3 = Add()([upsample3, conv1_1])  # Skip connection

    # Output Layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax', padding='same')(skip3)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model