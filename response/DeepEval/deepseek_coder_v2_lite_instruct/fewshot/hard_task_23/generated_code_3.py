import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Conv2DTranspose

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First branch: Local feature extraction
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    # Second branch: Downsampling followed by 3x3 convolutional and transposed convolutional layer
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch2)

    # Third branch: Similar to second branch but uses different pooling and upsampling
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch3)

    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Refine the concatenated output with a 1x1 convolutional layer
    refined = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Flatten and pass through fully connected layer for classification
    flattened = Flatten()(refined)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model