import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Define the block that contains three sequential convolutional layers
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3
    
    # First branch
    block_output1 = block(input_layer)
    
    # Second branch (same block)
    block_output2 = block(input_layer)
    
    # Parallel branch connecting input directly through a convolutional layer
    direct_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine all outputs from the three branches using Add
    added_output = Add()([block_output1, block_output2, direct_conv])
    
    # Concatenate the outputs from the two blocks
    concatenated_output = Concatenate()([block_output1, block_output2])
    
    # Pass through flattening layer
    flatten_layer = Flatten()(concatenated_output)
    
    # Fully connected layer for classification
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model