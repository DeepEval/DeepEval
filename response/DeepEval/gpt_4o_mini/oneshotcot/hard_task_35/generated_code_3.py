import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    def block(input_tensor):
        # Global Average Pooling
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        
        # Two Fully Connected Layers
        dense1 = Dense(units=128, activation='relu')(global_avg_pool)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)  # Output dimension matches channels
        
        # Reshape the weights to the shape of the input
        reshaped_weights = Reshape((1, 1, input_tensor.shape[-1]))(dense2)  # Reshape to (1, 1, channels)
        
        # Element-wise multiplication with the input
        scaled_output = Multiply()([input_tensor, reshaped_weights])
        
        return scaled_output
    
    # First branch
    branch1_output = block(input_layer)
    
    # Second branch (identical structure)
    branch2_output = block(input_layer)
    
    # Concatenate outputs from both branches
    concatenated = Concatenate()([branch1_output, branch2_output])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()  # To check the structure of the model