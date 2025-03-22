from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Part 1: Feature extraction
    # First pair: Conv -> MaxPooling
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Second pair: Conv -> MaxPooling
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Third pair: Conv -> MaxPooling
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # Part 2: Enhance generalization
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    drop4 = Dropout(0.5)(conv4)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(drop4)

    # Part 3: Upsampling with skip connections
    # First pair: Conv -> TransposedConv
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    up1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    skip1 = concatenate([up1, conv3])

    # Second pair: Conv -> TransposedConv
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(skip1)
    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    skip2 = concatenate([up2, conv2])

    # Third pair: Conv -> TransposedConv
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(skip2)
    up3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    skip3 = concatenate([up3, conv1])

    # Output layer: 1x1 Convolution for class predictions
    outputs = Conv2D(10, (1, 1), activation='softmax', padding='same')(skip3)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model