import keras
from keras.layers import Input, Conv2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Define the first block
    def block(input_tensor):
        # First convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Second convolutional layer
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        # Third convolutional layer
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        
        # Separate paths for each convolutional layer
        path1 = conv1
        path2 = conv2
        path3 = conv3
        
        # Parallel branch starting from the input
        parallel_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Combine outputs using addition
        output_tensor = Add()([path1, path2, path3, parallel_path])
        
        return output_tensor
    
    # Apply the block to the input
    block1_output = block(input_layer)
    
    # Repeat the block for the second parallel branch
    block2_output = block(input_layer)
    
    # Concatenate the outputs from the two blocks
    concatenated_output = Concatenate()([block1_output, block2_output])
    
    # Flatten the concatenated output
    flattened_output = Flatten()(concatenated_output)
    
    # Pass through a fully connected layer
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    
    # Final classification output
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model