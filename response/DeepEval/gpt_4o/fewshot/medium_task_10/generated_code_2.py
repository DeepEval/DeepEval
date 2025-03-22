import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor, filters):
    # Main path
    conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)
    
    # Branch path (identity connection)
    branch = input_tensor
    
    # Add both paths
    output = Add()([relu, branch])
    return output

def residual_block(input_tensor, filters):
    # Main path using the basic block
    main_path = basic_block(input_tensor, filters=filters)
    
    # Branch path with independent convolutional layer
    branch = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
    
    # Combine paths
    output = Add()([main_path, branch])
    return output

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Adjust dimensionality of input features to 16
    conv_initial = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    # First level: single basic block
    level1_output = basic_block(conv_initial, filters=16)
    
    # Second level: two residual blocks
    level2_block1 = residual_block(level1_output, filters=32)
    level2_block2 = residual_block(level2_block1, filters=32)
    
    # Third level: global branch with convolution capturing initial conv output
    global_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_initial)
    
    # Combine second level output with third level global branch
    level3_output = Add()([level2_block2, global_branch])
    
    # Average pooling and fully connected layer for classification
    avg_pool = AveragePooling2D(pool_size=(8, 8))(level3_output)
    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model