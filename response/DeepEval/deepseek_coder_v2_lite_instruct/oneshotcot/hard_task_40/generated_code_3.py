import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, AveragePooling2D, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    
    # Flatten and Concatenate the outputs
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Reshape the concatenated output into a 4D tensor
    reshape_layer = Reshape((1, 1, 3 * 7 * 7))(concatenated)
    
    # Second Block
    def second_block(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        
        # Path 2: 1x1 -> 3x3 -> 3x3 convolution
        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path2)
        
        # Path 3: 1x1 -> 3x3 convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path3)
        
        # Path 4: 1x1 convolution -> average pooling
        path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(path4)
        
        # Concatenate outputs from all paths
        output_tensor = Concatenate(axis=-1)([path1, path2, path3, path4])
        
        # Dropout for regularization
        output_tensor = Dropout(0.5)(output_tensor)
        
        return output_tensor
    
    second_block_output = second_block(reshape_layer)
    
    # Flatten the output from the second block
    flatten_layer = Flatten()(second_block_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model