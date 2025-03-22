import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, Conv2D

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        
        # Flatten and apply Dropout
        flat1 = Dropout(0.3)(Flatten()(path1))
        flat2 = Dropout(0.3)(Flatten()(path2))
        flat3 = Dropout(0.3)(Flatten()(path3))
        
        # Concatenate the flattened outputs
        output_tensor = Concatenate()([flat1, flat2, flat3])
        
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Fully connected layer and Reshape operation
    fc1 = Dense(units=7 * 7 * 8, activation='relu')(block1_output)  # Assuming a shape of (7, 7, 8) for reshaping
    reshaped = Reshape(target_shape=(7, 7, 8))(fc1)
    
    # Block 2
    def block2(input_tensor):
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch2)
        
        branch3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch3)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch3)
        
        branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(branch4)
        
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        
        return output_tensor
    
    block2_output = block2(reshaped)
    
    # Flatten and final fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model