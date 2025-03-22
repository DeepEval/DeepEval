import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First part: Convolutional layers with max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Second part: Convolutional layers with dropout
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    drop1 = Dropout(0.5)(conv4)  # Dropout for regularization
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(drop1)

    # Third part: Upsampling with skip connections
    up1 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv5)
    skip1 = Concatenate()([up1, conv3])  # Skip connection
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(skip1)

    up2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv6)
    skip2 = Concatenate()([up2, conv2])  # Skip connection
    conv7 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(skip2)

    up3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv7)
    skip3 = Concatenate()([up3, conv1])  # Skip connection
    conv8 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(skip3)

    # Final output layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv8)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model