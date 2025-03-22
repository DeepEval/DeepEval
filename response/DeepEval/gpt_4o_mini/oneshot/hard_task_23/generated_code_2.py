import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32x3

    # Initial 1x1 convolution
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: Local feature extraction with two 3x3 conv layers
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    # Branch 2: Average pooling followed by a 3x3 conv and transposed conv
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch2)

    # Branch 3: Similar to Branch 2
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch3)

    # Concatenate the outputs of the branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Apply a 1x1 convolution to refine the concatenated output
    refined_output = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Flatten and pass through fully connected layer
    flatten_layer = Flatten()(refined_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model