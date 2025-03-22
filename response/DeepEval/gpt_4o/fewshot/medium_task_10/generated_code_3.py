import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Adjust input feature dimensionality to 16
    conv_initial = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Basic block definition
    def basic_block(input_tensor):
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        return relu
    
    # Level 1: Single basic block
    level1_main_path = basic_block(conv_initial)
    level1_output = Add()([level1_main_path, conv_initial])

    # Level 2: Two residual blocks
    def residual_block(input_tensor):
        main_path = basic_block(input_tensor)
        branch_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        return Add()([main_path, branch_path])

    level2_residual1 = residual_block(level1_output)
    level2_residual2 = residual_block(level2_residual1)
    
    # Level 3: Global branch captures features from initial convolution output
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_initial)
    level3_output = Add()([level2_residual2, global_branch])

    # Average pooling and fully connected layer for classification
    avg_pooling = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')(level3_output)
    flatten = Flatten()(avg_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model