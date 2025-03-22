import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten
from keras.models import Model

def block(input_tensor):
    # Apply global average pooling
    gap = GlobalAveragePooling2D()(input_tensor)
    # First dense layer to generate weights
    dense1 = Dense(units=gap.shape[-1], activation='relu')(gap)
    # Second dense layer to match the channel size
    dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
    # Reshape to match the input dimensions except batch size
    weights = Reshape((1, 1, input_tensor.shape[-1]))(dense2)
    # Element-wise multiplication with the input
    scaled_input = Multiply()([input_tensor, weights])
    return scaled_input

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1
    branch1 = block(input_layer)
    
    # Branch 2
    branch2 = block(input_layer)
    
    # Concatenate outputs from both branches
    concatenated_output = Concatenate()([branch1, branch2])
    
    # Flatten the concatenated output
    flattened_output = Flatten()(concatenated_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened_output)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example of constructing the model
model = dl_model()
model.summary()