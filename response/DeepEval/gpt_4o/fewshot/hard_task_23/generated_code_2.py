import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer
    initial_conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First branch: Local feature extraction with two sequential 3x3 convolutional layers
    branch1_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    branch1_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1_conv1)

    # Second branch: Average pooling, 3x3 convolution, and transposed convolution
    branch2_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch2_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)
    branch2_upsample = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch2_conv)

    # Third branch: Similar to the second branch
    branch3_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch3_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_upsample = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch3_conv)

    # Concatenate outputs from all three branches
    concatenated = Concatenate()([branch1_conv2, branch2_upsample, branch3_upsample])

    # Refine using a 1x1 convolutional layer
    refined = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Fully connected layer for classification
    flatten = Flatten()(refined)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model