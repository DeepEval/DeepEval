import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Conv2D, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path1_flatten = Flatten()(path1)
        
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path2_flatten = Flatten()(path2)
        
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        path3_flatten = Flatten()(path3)
        
        output_tensor = Concatenate()([path1_flatten, path2_flatten, path3_flatten])
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Fully connected layer between blocks
    fc1 = Dense(units=128, activation='relu')(block1_output)
    # Reshape into 4D tensor for block 2
    reshaped = Reshape((4, 4, 8))(fc1)  # Assumes the Dense layer output size can be reshaped to (4, 4, 8)
    
    # Block 2
    def block2(input_tensor):
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch1)
        
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu', padding='same')(branch2)
        branch2 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(branch2)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch2)
        
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor
    
    block2_output = block2(reshaped)
    
    # Final Dense layers for classification
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model