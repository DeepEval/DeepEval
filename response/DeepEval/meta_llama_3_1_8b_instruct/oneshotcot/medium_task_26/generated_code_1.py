import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Step 2: Compress the input channels with a 1x1 convolutional layer
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Define a block with two parallel convolutional layers
    def block(input_tensor):
        
        # Step 3.1: Add 1x1 convolutional layer as the first path
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Step 3.2: Add 3x3 convolutional layer as the second path
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Step 3.3: Concatenate the outputs of the two paths
        output_tensor = Concatenate()([path1, path2])
        
        return output_tensor
    
    # Step 4: Apply the block to the output of the 1x1 convolutional layer
    block_output = block(conv)
    
    # Step 5: Add batch normalization layer
    bath_norm = BatchNormalization()(block_output)
    
    # Step 6: Flatten the output feature map
    flatten_layer = Flatten()(bath_norm)
    
    # Step 7: Add the first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 8: Add the second fully connected layer
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Step 9: Create the model
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model