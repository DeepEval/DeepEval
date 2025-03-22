import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Concatenate, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        # Global average pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        
        # Two fully connected layers
        dense1 = Dense(units=64, activation='relu')(gap)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
        
        # Reshape weights to match input shape
        weights = keras.layers.Reshape((1, 1, input_tensor.shape[-1]))(dense2)
        
        # Element-wise multiplication with input
        scaled_tensor = Multiply()([input_tensor, weights])
        
        return scaled_tensor

    # Two branches using the same block
    branch1 = block(input_layer)
    branch2 = block(input_layer)

    # Concatenate outputs from both branches
    concatenated = Concatenate()([branch1, branch2])
    
    # Flatten the result and add a dense layer for classification
    flatten_layer = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model