from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

def dl_model():
    inputs = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    # Part 1: Feature Extraction
    # First convolutional block
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Second convolutional block
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Third convolutional block
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # Part 2: Generalization enhancement
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    drop4 = Dropout(0.5)(conv4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(drop4)

    # Part 3: Upsampling with skip connections
    # First upsampling block
    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = concatenate([up6, conv3], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge6)

    # Second upsampling block
    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([up7, conv2], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge7)

    # Third upsampling block
    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([up8, conv1], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge8)

    # Output layer with 1x1 convolution
    outputs = Conv2D(10, (1, 1), activation='softmax')(conv8)

    model = Model(inputs=inputs, outputs=outputs)

    return model