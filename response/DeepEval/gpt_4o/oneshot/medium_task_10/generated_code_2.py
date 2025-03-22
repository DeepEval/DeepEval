import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Adjust input feature dimensionality to 16 using a convolutional layer
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Basic Block
    def basic_block(input_tensor):
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = ReLU()(main_path)
        
        # Adding the main path to the branch (identity)
        output_tensor = Add()([main_path, input_tensor])
        return output_tensor

    # Level 1: Single Basic Block
    level1_output = basic_block(initial_conv)
    
    # Level 2: Two Residual Blocks
    def residual_block(input_tensor):
        main_path = basic_block(input_tensor)
        
        # Branch with independent conv layer
        branch_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        
        # Adding main path to branch path
        output_tensor = Add()([main_path, branch_path])
        return output_tensor
    
    level2_output_1 = residual_block(level1_output)
    level2_output_2 = residual_block(level2_output_1)
    
    # Level 3: Convolutional layer in the global branch
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(initial_conv)
    
    # Adding global branch to Level 2 output
    level3_output = Add()([level2_output_2, global_branch])
    
    # Average Pooling followed by Fully Connected Layer
    avg_pool = AveragePooling2D(pool_size=(8, 8))(level3_output)
    flatten_layer = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model