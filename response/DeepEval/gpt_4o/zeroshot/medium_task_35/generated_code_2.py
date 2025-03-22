from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    inputs = Input(shape=(32, 32, 3))
    
    # First stage of convolution and max pooling
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Second stage of convolution and max pooling
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Additional convolutional and dropout layers
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.5)(conv3)

    # Upsampling and skip connections
    up1 = UpSampling2D(size=(2, 2))(conv3)
    skip1 = concatenate([up1, conv2], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(skip1)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    skip2 = concatenate([up2, conv1], axis=-1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(skip2)

    # Final 1x1 convolutional layer for classification
    outputs = Conv2D(10, (1, 1), activation='softmax')(conv5)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model