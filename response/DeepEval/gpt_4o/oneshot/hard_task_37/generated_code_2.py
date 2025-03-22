import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        # First sequential block
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
        
        # Second sequential block
        conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
        conv6 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv5)
        
        # Parallel path
        parallel_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        
        # Add operation
        add_path1 = Add()([conv3, parallel_conv])
        add_path2 = Add()([conv6, parallel_conv])
        
        return add_path1, add_path2
    
    # First block
    block1_output1, block1_output2 = block(input_tensor=input_layer)
    
    # Second block
    block2_output1, block2_output2 = block(input_tensor=input_layer)
    
    # Concatenation of two blocks' outputs
    concat_output = Concatenate()([block1_output1, block1_output2, block2_output1, block2_output2])
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(concat_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model