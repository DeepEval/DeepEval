import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Reshape, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    
    # Flatten the outputs and concatenate
    concat_layer = Concatenate(axis=-1)([Flatten()(pool1), Flatten()(pool2), Flatten()(pool3)])
    
    # Reshape the concatenated output to a 4D tensor
    reshape_layer = Reshape((4, 1, 1))(concat_layer)
    
    # Second block
    def second_block(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        
        # Path 2: 1x1 followed by two 3x3 convolutions
        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path2)
        
        # Path 3: 1x1 followed by a single 3x3 convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path3)
        
        # Path 4: 1x1 convolution followed by average pooling
        path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(path4)
        
        # Concatenate outputs from all paths
        output_tensor = Concatenate(axis=-1)([path1, path2, path3, path4])
        
        # Add dropout for regularization
        output_tensor = Dropout(0.5)(output_tensor)
        
        return output_tensor
    
    block_output = second_block(reshape_layer)
    
    # Flatten the output from the second block
    flatten_layer = Flatten()(block_output)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model