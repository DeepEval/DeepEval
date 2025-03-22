import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Define the block structure
    def block(input_tensor):
        # Sequential convolutional layers
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
        
        # Parallel branch connecting the input through a convolutional layer
        parallel_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

        # Combine the outputs using an addition operation
        added = Add()([conv3, parallel_conv])
        
        return added
    
    # Two parallel branches, each with the same block
    block1_output = block(input_layer)
    block2_output = block(input_layer)
    
    # Concatenate the outputs from the two blocks
    concatenated = Concatenate()([block1_output, block2_output])
    
    # Flattening layer
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model