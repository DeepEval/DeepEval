import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Concatenate, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        # Apply global average pooling to compress input features
        global_pool = GlobalAveragePooling2D()(input_tensor)
        
        # Pass pooled output through two fully connected layers
        dense1 = Dense(units=64, activation='relu')(global_pool)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
        
        # Reshape weights to match the input shape
        reshaped_weights = keras.layers.Reshape((1, 1, input_tensor.shape[-1]))(dense2)
        
        # Element-wise multiply the weights with the block's input
        scaled_input = Multiply()([input_tensor, reshaped_weights])
        
        return scaled_input

    # First branch
    branch1 = block(input_layer)
    
    # Second branch
    branch2 = block(input_layer)
    
    # Concatenate outputs from both branches
    concatenated = Concatenate(axis=-1)([branch1, branch2])
    
    # Flattening layer
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model