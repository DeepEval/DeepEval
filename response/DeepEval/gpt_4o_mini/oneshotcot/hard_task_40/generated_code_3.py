import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Dropout
from keras.models import Model

def dl_model():
    # Step 1: Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: First block with average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flattening the outputs of the average pooling layers
    flat1 = Flatten()(avg_pool1)
    flat2 = Flatten()(avg_pool2)
    flat3 = Flatten()(avg_pool3)

    # Concatenating the flattened outputs
    concatenated = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layer
    fc1 = Dense(units=128, activation='relu')(concatenated)
    
    # Reshape the output into a 4D tensor (batch_size, 1, 1, 128)
    reshaped = Reshape((1, 1, 128))(fc1)

    # Step 3: Second block with parallel paths
    def second_block(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        path1 = Dropout(0.5)(path1)

        # Path 2: Two 3x3 convolutions stacked after a 1x1 convolution
        path2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path2)
        path2 = Dropout(0.5)(path2)

        # Path 3: One 3x3 convolution after a 1x1 convolution
        path3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path3)
        path3 = Dropout(0.5)(path3)

        # Path 4: Average pooling with a 1x1 convolution
        path4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1))(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(path4)
        path4 = Dropout(0.5)(path4)

        # Concatenate the outputs from all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor
    
    # Applying the second block
    block_output = second_block(reshaped)

    # Flatten the output for fully connected layers
    flat_block_output = Flatten()(block_output)

    # Final fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat_block_output)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Build the model
    model = Model(inputs=input_layer, outputs=dense2)
    
    return model