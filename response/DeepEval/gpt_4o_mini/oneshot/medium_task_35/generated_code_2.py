import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # First convolutional stage
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Second convolutional stage
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Additional convolutional and dropout layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    conv3 = Dropout(0.5)(conv3)  # Adding dropout for regularization
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv3)

    # Upsampling and skip connections
    up1 = UpSampling2D(size=(2, 2))(conv3)
    skip1 = Concatenate()([up1, conv2])  # Skip connection
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(skip1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    skip2 = Concatenate()([up2, conv1])  # Skip connection
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(skip2)
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv5)

    # Final layer for classification
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv5)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model