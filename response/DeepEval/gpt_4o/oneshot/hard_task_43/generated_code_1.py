import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Average Pooling with different scales
    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        
        # Flatten the outputs
        path1_flat = Flatten()(path1)
        path2_flat = Flatten()(path2)
        path3_flat = Flatten()(path3)
        
        # Concatenate flattened outputs
        output_tensor = Concatenate()([path1_flat, path2_flat, path3_flat])
        
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Fully connected layer between block 1 and block 2
    fc1 = Dense(units=256, activation='relu')(block1_output)
    
    # Reshape for block 2 processing, assuming we want a 4D tensor of shape suitable for convolutions
    reshaped = Reshape((4, 4, 16))(fc1)  # Example reshape, adjust dimensions based on your architecture needs
    
    # Block 2: Feature extraction with different branches
    def block2(input_tensor):
        # Branch 1: 1x1 convolution followed by 3x3 convolution
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch1)
        
        # Branch 2: 1x1, 1x7, 7x1, and 3x3 convolutions
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3: Average pooling
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate outputs
        output_tensor = Concatenate()([branch1, branch2, branch3])
        
        return output_tensor
    
    block2_output = block2(reshaped)
    
    # Final fully connected layers for classification
    flatten = Flatten()(block2_output)
    fc2 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(fc2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model