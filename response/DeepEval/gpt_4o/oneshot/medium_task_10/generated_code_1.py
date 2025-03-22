import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to adjust input feature dimensionality to 16
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Basic Block
    def basic_block(input_tensor):
        # Main Path
        main_path = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = ReLU()(main_path)

        # Branch (Identity)
        branch = input_tensor
        
        # Add main path and branch
        block_output = Add()([main_path, branch])

        return block_output

    # First Level: Single Basic Block
    level1_output = basic_block(initial_conv)

    # Second Level: Two Residual Blocks
    def residual_block(input_tensor):
        # Main Path using Basic Block
        main_path = basic_block(input_tensor)

        # Branch with independent Conv Layer
        branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_tensor)
        
        # Add main path and branch
        block_output = Add()([main_path, branch])
        
        return block_output

    # Two residual blocks
    level2_output = residual_block(level1_output)
    level2_output = residual_block(level2_output)

    # Third Level: Feature Fusion with Global Convolution
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(initial_conv)

    # Add global branch with second level output
    level3_output = Add()([level2_output, global_branch])

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(level3_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(gap)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model