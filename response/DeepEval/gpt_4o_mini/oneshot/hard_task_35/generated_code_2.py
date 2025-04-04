import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    def block(input_tensor):
        # Global Average Pooling
        pooled_output = GlobalAveragePooling2D()(input_tensor)
        
        # Fully connected layers to produce weights
        dense1 = Dense(units=128, activation='relu')(pooled_output)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)  # Output size equal to the number of channels
        
        # Reshape weights to match input shape
        reshaped_weights = Reshape((1, 1, input_tensor.shape[-1]))(dense2)

        # Element-wise multiplication with the input
        scaled_output = Multiply()([input_tensor, reshaped_weights])
        
        return scaled_output

    # First branch
    branch1_output = block(input_layer)
    # Second branch with the same block
    branch2_output = block(input_layer)

    # Concatenate the outputs from both branches
    concatenated_output = Concatenate()([branch1_output, branch2_output])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated_output)
    
    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model