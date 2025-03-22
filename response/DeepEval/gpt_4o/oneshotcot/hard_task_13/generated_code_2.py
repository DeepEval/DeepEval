import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Multiply, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block with four parallel branches
    def first_block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate the outputs of the four branches
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor
    
    block1_output = first_block(input_layer)
    
    # Second block with global average pooling and channel-wise weighting
    def second_block(input_tensor):
        pooled = GlobalAveragePooling2D()(input_tensor)
        
        # Two fully connected layers to generate channel-wise weights
        dense1 = Dense(units=128, activation='relu')(pooled)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
        
        # Reshape the dense output to match input dimensions
        weights = Reshape((1, 1, input_tensor.shape[-1]))(dense2)
        
        # Element-wise multiplication with the input feature map
        weighted_features = Multiply()([input_tensor, weights])
        
        return weighted_features
    
    block2_output = second_block(block1_output)
    
    # Final fully connected layer to produce the output
    output_layer = Dense(units=10, activation='softmax')(GlobalAveragePooling2D()(block2_output))
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model