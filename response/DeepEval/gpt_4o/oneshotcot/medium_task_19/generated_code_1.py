import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def branch_block(input_tensor):
        # First branch: 1x1 convolution for dimensionality reduction
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Second branch: 1x1 convolution followed by a 3x3 convolution
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Third branch: 1x1 convolution followed by a 5x5 convolution
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(branch3)
        
        # Fourth branch: 3x3 max pooling followed by a 1x1 convolution
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch4)

        # Concatenate all branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor

    # Apply the block to the input layer
    block_output = branch_block(input_layer)

    # Flatten the concatenated features
    flatten_layer = Flatten()(block_output)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model