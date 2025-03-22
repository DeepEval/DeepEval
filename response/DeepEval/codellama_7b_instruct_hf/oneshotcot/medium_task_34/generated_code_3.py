from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Flatten, Dense, BatchNormalization
from keras.models import Model

def cifar10_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First part: feature extraction
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu')(max_pool2)
    max_pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Second part: feature enhancement
    conv4 = Conv2D(128, (3, 3), activation='relu')(max_pool3)
    dropout1 = Dropout(0.5)(conv4)
    conv5 = Conv2D(128, (3, 3), activation='relu')(dropout1)

    conv6 = Conv2D(128, (3, 3), activation='relu')(conv5)
    dropout2 = Dropout(0.5)(conv6)
    conv7 = Conv2D(128, (3, 3), activation='relu')(dropout2)

    # Third part: upsampling
    up1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv7)
    up1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(up1)
    up1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(up1)
    skip1 = Conv2D(64, (3, 3), activation='relu')(conv4)
    up1 = Concatenate()([up1, skip1])

    up2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(up1)
    up2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(up2)
    up2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(up2)
    skip2 = Conv2D(32, (3, 3), activation='relu')(conv3)
    up2 = Concatenate()([up2, skip2])

    up3 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(up2)
    up3 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(up3)
    up3 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(up3)
    skip3 = Conv2D(16, (3, 3), activation='relu')(conv2)
    up3 = Concatenate()([up3, skip3])

    # 1x1 convolutional layer for class probabilities
    output_layer = Conv2D(10, (1, 1), activation='softmax')(up3)

    # Define and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model