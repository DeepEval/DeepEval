import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 input shape

    # Initial 1x1 convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)

    # Branch 1: Local feature extraction through a 3x3 convolutional layer
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(initial_conv)

    # Branch 2: Max pooling, 3x3 convolution, and upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2), padding='same')(initial_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: Max pooling, 3x3 convolution, and upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2), padding='same')(initial_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenating the outputs of the branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Another 1x1 convolutional layer after concatenation
    post_concat_conv = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', padding='same')(concatenated)

    # Flatten and pass through fully connected layers
    flatten = Flatten()(post_concat_conv)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model