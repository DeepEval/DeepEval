import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor):
        # Global Average Pooling to compress the input features
        gap = GlobalAveragePooling2D()(input_tensor)
        
        # Two fully connected layers to produce weights
        dense1 = Dense(units=128, activation='relu')(gap)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
        
        # Reshape weights to match input shape
        reshaped_weights = Reshape((1, 1, input_tensor.shape[-1]))(dense2)
        
        # Element-wise multiplication with the input
        scaled_input = Multiply()([input_tensor, reshaped_weights])
        
        return scaled_input
    
    # Create two branches using the block
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    
    # Concatenate the outputs from both branches
    concatenated = Concatenate()([branch1, branch2])
    
    # Flatten the concatenated result
    flatten_layer = Flatten()(concatenated)
    
    # Add a fully connected layer for the classification output
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model