import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Reshape, Concatenate, Flatten

def dl_model():
    def block(input_tensor):
        # Apply global average pooling
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        # Fully connected layers to produce weights
        dense1 = Dense(units=64, activation='relu')(global_avg_pool)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
        # Reshape and multiply with the input tensor
        reshaped_weights = Reshape((1, 1, input_tensor.shape[-1]))(dense2)
        weighted_input = Multiply()([input_tensor, reshaped_weights])
        return weighted_input

    # Input layer for CIFAR-10 (32x32 RGB images)
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch
    branch1_output = block(input_layer)
    
    # Second branch
    branch2_output = block(input_layer)
    
    # Concatenate outputs from both branches
    concatenated_output = Concatenate()([branch1_output, branch2_output])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model