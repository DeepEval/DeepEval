import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Adjust input dimensionality to 16 using a convolutional layer
    conv1 = Conv2D(16, (3, 3), activation='relu')(input_layer)
    
    # Define a basic block
    def basic_block(input_tensor):
        # Main path: Conv2D, BatchNormalization, ReLU
        conv = Conv2D(16, (3, 3), activation='relu')(input_tensor)
        bn = BatchNormalization()(conv)
        return bn
    
    # First path of the basic block
    block_path1 = basic_block(input_tensor=conv1)
    
    # Branch to the basic block
    branch = Conv2D(16, (1, 1), activation='relu')(conv1)
    
    # Combine outputs from both paths using an addition operation
    combined = Add()([block_path1, branch])
    
    # Second level residual block
    def second_level_block(input_tensor):
        # Main path: Basic block + branch
        block_path1 = basic_block(input_tensor)
        branch = Conv2D(16, (1, 1), activation='relu')(input_tensor)
        combined = Add()([block_path1, branch])
        # Batch normalization and ReLU
        bn = BatchNormalization()(combined)
        return bn
    
    # Third level residual block
    def third_level_block(input_tensor):
        # Global branch convolution
        global_branch = Conv2D(16, (3, 3), activation='relu')(input_tensor)
        # Add main path and global branch
        second_level = second_level_block(input_tensor=input_tensor)
        combined = Add()([second_level, global_branch])
        # Batch normalization and ReLU
        bn = BatchNormalization()(combined)
        return bn
    
    # Apply the third level residual block
    final_output = third_level_block(input_tensor=conv1)
    
    # Map the features to classification probabilities
    avg_pool = GlobalAveragePooling2D()(final_output)
    output_layer = Dense(units=10, activation='softmax')(avg_pool)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()