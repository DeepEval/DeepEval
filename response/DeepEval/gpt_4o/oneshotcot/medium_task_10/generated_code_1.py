import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor, filters):
    # Main path
    conv = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=None)(input_tensor)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)
    
    # Branch (identity)
    # Just passing the input_tensor directly for an identity shortcut

    # Combine both paths
    output = Add()([relu, input_tensor])
    return output

def residual_block(input_tensor, filters):
    # Main path
    basic_block_output = basic_block(input_tensor, filters)
    
    # Branch
    branch_conv = Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(input_tensor)
    
    # Combine both paths
    output = Add()([basic_block_output, branch_conv])
    return output

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to adjust input feature dimensionality to 16
    conv_initial = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Level 1: Basic block
    level1_output = basic_block(conv_initial, filters=16)
    
    # Level 2: Two residual blocks
    level2_output = residual_block(level1_output, filters=32)
    level2_output = residual_block(level2_output, filters=32)
    
    # Level 3: Global branch
    global_branch_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(conv_initial)
    
    # Combine Level 2 and Global Branch
    level3_output = Add()([level2_output, global_branch_conv])
    
    # Average pooling followed by a fully connected layer for classification
    avg_pool = AveragePooling2D(pool_size=(8, 8))(level3_output)
    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model