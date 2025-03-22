import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Reshape, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Parallel Average Pooling with different scales
    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        
        flat1 = Flatten()(path1)
        flat2 = Flatten()(path2)
        flat3 = Flatten()(path3)
        
        output_tensor = Concatenate()([flat1, flat2, flat3])
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Fully connected layer after Block 1
    fc1 = Dense(units=256, activation='relu')(block1_output)
    
    # Reshape for Block 2
    reshaped = Reshape((4, 4, 16))(fc1)
    
    # Block 2: Feature Extraction with various configurations
    def block2(input_tensor):
        # Branch 1
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
        
        # Branch 2
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        # Branch 3
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
        
        # Concatenate outputs of all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor
    
    block2_output = block2(reshaped)
    
    # Flatten and fully connected layers for final classification
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model