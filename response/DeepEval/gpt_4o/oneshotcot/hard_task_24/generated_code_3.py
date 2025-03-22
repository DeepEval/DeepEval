import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)

    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(initial_conv)

    # Branch 2: MaxPooling -> Conv -> UpSampling
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: MaxPooling -> Conv -> UpSampling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate the outputs of the branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolutional layer to fuse concatenated features
    fusion_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(concatenated)

    # Flatten the output
    flatten_layer = Flatten()(fusion_conv)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model