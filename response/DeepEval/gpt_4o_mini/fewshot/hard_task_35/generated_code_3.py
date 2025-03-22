import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        # Global Average Pooling
        pooled_output = GlobalAveragePooling2D()(input_tensor)
        
        # Two Fully Connected Layers to produce weights
        dense1 = Dense(units=128, activation='relu')(pooled_output)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)  # Output dimension matches channel
        
        # Reshape to match input shape
        reshaped_weights = Reshape((1, 1, input_tensor.shape[-1]))(dense2)
        
        # Element-wise multiplication
        weighted_output = Multiply()([input_tensor, reshaped_weights])
        
        return weighted_output

    # Create two branches with the same block
    branch1_output = block(input_layer)
    branch2_output = block(input_layer)

    # Concatenate outputs from both branches
    concatenated_output = Concatenate()([branch1_output, branch2_output])

    # Flatten the concatenated output
    flatten_output = Flatten()(concatenated_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model