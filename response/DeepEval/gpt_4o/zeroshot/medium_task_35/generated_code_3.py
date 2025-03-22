import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # Encoder: Two stages of convolution and max pooling
    # Stage 1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Stage 2
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    # Decoder with skip connections and upsampling
    # Upsampling 1
    up1 = UpSampling2D((2, 2))(conv3)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    skip1 = concatenate([up1, conv2], axis=3)  # Skip connection
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(skip1)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    # Upsampling 2
    up2 = UpSampling2D((2, 2))(conv4)
    up2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    skip2 = concatenate([up2, conv1], axis=3)  # Skip connection
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(skip2)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    # Output layer with 1x1 convolution for 10 categories
    outputs = Conv2D(10, (1, 1), activation='softmax')(conv5)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model