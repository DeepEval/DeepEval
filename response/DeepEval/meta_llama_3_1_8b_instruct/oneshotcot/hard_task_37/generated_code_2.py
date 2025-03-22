import keras
from keras.layers import Input, Conv2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Define a block
    def block(input_tensor):
        
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        
        # Create parallel paths
        path1 = conv3
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Add the outputs of the paths
        output_tensor = Add()([path1, path2])
        
        return output_tensor
    
    # Create two blocks
    block_output1 = block(input_layer)
    block_output2 = block(block_output1)
    
    # Concatenate the outputs of the two blocks
    output_tensor = Concatenate()([block_output1, block_output2])
    
    # Add a batch normalization layer
    bath_norm = BatchNormalization()(output_tensor)
    
    # Add a flatten layer
    flatten_layer = Flatten()(bath_norm)
    
    # Add a dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Add a final dense layer
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model