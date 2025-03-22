import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Conv2DTranspose
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First branch: local feature extraction
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    # Second branch: downsampling followed by upsampling
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch2)

    # Third branch: downsampling followed by upsampling
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch3)

    # Concatenate outputs of the branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolutional layer to refine the output
    refined = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Flatten the result
    flattened = Flatten()(refined)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flattened)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model