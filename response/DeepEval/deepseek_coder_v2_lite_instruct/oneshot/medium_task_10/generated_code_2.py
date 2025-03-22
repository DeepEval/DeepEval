import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution to adjust the input feature dimensionality to 16
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = ReLU()(batch_norm1)
    
    # Define the basic block
    def basic_block(input_tensor):
        # Main path
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm_main = BatchNormalization()(main_path)
        relu_main = ReLU()(batch_norm_main)
        
        # Branch path
        branch = input_tensor
        
        # Add the main path and branch path
        output_tensor = Add()([relu_main, branch])
        
        return output_tensor
    
    # First level of the residual structure
    block1 = basic_block(relu1)
    
    # Second level of the residual structure (two blocks)
    block2_1 = basic_block(block1)
    block2_2 = basic_block(block2_1)
    
    # Third level of the residual structure
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    global_branch = BatchNormalization()(global_branch)
    global_branch = ReLU()(global_branch)
    
    # Add the global branch to the second-level residual structure
    final_block = Add()([block2_2, global_branch])
    
    # Average pooling and fully connected layer
    avg_pool = AveragePooling2D(pool_size=(4, 4))(final_block)
    flatten_layer = Flatten()(avg_pool)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=dense_layer)
    
    return model