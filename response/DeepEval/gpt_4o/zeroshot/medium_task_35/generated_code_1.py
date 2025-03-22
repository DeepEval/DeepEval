import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First stage of convolution and pooling
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Second stage of convolution and pooling
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Additional convolutional and dropout layers
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    dropout1 = Dropout(0.5)(conv3)

    # Upsampling and skip connections
    up1 = UpSampling2D(size=(2, 2))(dropout1)
    up1 = Conv2D(64, (2, 2), activation='relu', padding='same')(up1)
    skip1 = concatenate([up1, conv2], axis=3)

    up2 = UpSampling2D(size=(2, 2))(skip1)
    up2 = Conv2D(32, (2, 2), activation='relu', padding='same')(up2)
    skip2 = concatenate([up2, conv1], axis=3)

    # Output layer
    outputs = Conv2D(10, (1, 1), activation='softmax')(skip2)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model