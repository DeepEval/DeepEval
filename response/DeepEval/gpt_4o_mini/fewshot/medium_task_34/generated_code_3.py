import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Part 1: Feature extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Part 2: Enhance generalization capabilities
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    dropout = Dropout(rate=0.5)(conv4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(dropout)

    # Part 3: Upsampling with skip connections
    upconv1 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5)
    skip1 = Concatenate()([upconv1, conv3])  # Skip connection
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(skip1)

    upconv2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6)
    skip2 = Concatenate()([upconv2, conv2])  # Skip connection
    conv7 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(skip2)

    upconv3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    skip3 = Concatenate()([upconv3, conv1])  # Skip connection
    conv8 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(skip3)

    # Final output layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv8)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model