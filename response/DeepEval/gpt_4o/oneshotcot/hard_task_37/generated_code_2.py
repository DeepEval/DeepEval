import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    
    def block(input_tensor):
        # Three sequential convolutional layers
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
        
        # Parallel branch with a convolutional layer
        parallel_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Add the outputs of sequential layers and the parallel branch
        added_output = Add()([conv1, conv2, conv3, parallel_conv])
        
        return added_output

    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    block1_output = block(input_layer)
    
    # Block 2
    block2_output = block(input_layer)
    
    # Concatenate the outputs of the two blocks
    concatenated_output = Concatenate()([block1_output, block2_output])
    
    # Flatten layer
    flatten_layer = Flatten()(concatenated_output)
    
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model