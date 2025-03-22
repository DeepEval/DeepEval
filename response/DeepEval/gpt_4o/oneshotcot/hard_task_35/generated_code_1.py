import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Concatenate, Flatten
from keras.models import Model

def dl_model():
    def block(input_tensor):
        # Apply Global Average Pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        
        # First fully connected layer
        dense1 = Dense(units=input_tensor.shape[-1], activation='relu')(gap)
        
        # Second fully connected layer
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
        
        # Reshape weights to match input shape
        weights = keras.layers.Reshape((1, 1, input_tensor.shape[-1]))(dense2)
        
        # Element-wise multiplication with the input tensor
        scaled_tensor = Multiply()([input_tensor, weights])
        
        return scaled_tensor

    input_layer = Input(shape=(32, 32, 3))

    # First branch
    branch1 = block(input_layer)

    # Second branch
    branch2 = block(input_layer)

    # Concatenate outputs from both branches
    concatenated = Concatenate()([branch1, branch2])
    
    # Flatten the concatenated tensor
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model