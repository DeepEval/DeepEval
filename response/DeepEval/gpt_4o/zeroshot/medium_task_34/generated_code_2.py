from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from keras.models import Model

def dl_model():
    inputs = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Part 1: Feature extraction
    # Block 1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Block 2
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Block 3
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # Part 2: Generalization enhancement
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    dropout = Dropout(0.5)(conv4)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(dropout)

    # Part 3: Upsampling with skip connections
    up1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv5)
    skip1 = concatenate([up1, conv3], axis=3)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(skip1)

    up2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv6)
    skip2 = concatenate([up2, conv2], axis=3)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(skip2)

    up3 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv7)
    skip3 = concatenate([up3, conv1], axis=3)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(skip3)

    # Output layer
    outputs = Conv2D(10, (1, 1), activation='softmax')(conv8)

    # Construct model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating the model
model = dl_model()
model.summary()