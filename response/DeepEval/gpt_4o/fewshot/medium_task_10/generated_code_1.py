import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Adjust input feature dimensionality to 16
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_layer)

    # Define a basic block
    def basic_block(input_tensor):
        # Main path
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        # Addition of main and branch paths
        return Add()([input_tensor, x])

    # Level 1: Single basic block
    level_1_output = basic_block(initial_conv)

    # Level 2: Two residual blocks
    def residual_block(input_tensor):
        # Main path
        main_path = basic_block(input_tensor)
        
        # Branch path
        branch_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(input_tensor)
        
        # Addition of main and branch paths
        return Add()([main_path, branch_path])

    level_2_output_1 = residual_block(level_1_output)
    level_2_output_2 = residual_block(level_2_output_1)

    # Level 3: Global branch captures features
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(initial_conv)

    # Add global branch to level 2 output
    level_3_output = Add()([level_2_output_2, global_branch])

    # Average pooling and final fully connected layer for classification
    avg_pool = AveragePooling2D(pool_size=(8, 8))(level_3_output)
    flatten_layer = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model