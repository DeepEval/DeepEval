import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 2: Define the block with four parallel convolutional paths
    def block(input_tensor):
        # First path: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Second path: 1x1 convolution followed by two 3x3 convolutions
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)

        # Third path: 1x1 convolution followed by a 3x3 convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)

        # Fourth path: max pooling followed by a 1x1 convolution
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

        # Step 4.5: Concatenate the outputs of the paths
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor

    # Apply the block to the input
    block_output = block(input_tensor=input_layer)

    # Step 6: Add flatten layer
    flatten_layer = Flatten()(block_output)

    # Step 7: Add dense layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)

    # Step 8: Add the output layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model