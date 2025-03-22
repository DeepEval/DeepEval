import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Multiply

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Define the block
    def block(input_tensor):
        
        # Apply global average pooling to compress the input features
        pooling = GlobalAveragePooling2D()(input_tensor)
        
        # Pass the pooled output through two fully connected layers to produce weights
        dense1 = Dense(units=input_tensor.shape[-1], activation='relu')(pooling)
        dense2 = Dense(units=input_tensor.shape[-1], activation='relu')(dense1)
        
        # Reshape the weights to match the input's shape
        weights = Reshape((1, 1, input_tensor.shape[-1]))(dense2)
        
        # Element-wise multiply the weights with the block's input
        element_wise_multiply = Multiply()([input_tensor, weights])
        
        return element_wise_multiply
    
    # Define the two branches
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    
    # Concatenate the outputs from both branches
    concat = Concatenate()([branch1, branch2])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concat)
    
    # Add a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model