import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor, filters):
    # Main path
    conv = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
    norm = BatchNormalization()(conv)
    relu = ReLU()(norm)

    # Branch
    branch = input_tensor

    # Combine paths
    output_tensor = Add()([relu, branch])
    return output_tensor

def residual_block(input_tensor, filters):
    # Main path with basic block
    main_path = basic_block(input_tensor, filters=filters)

    # Branch with independent convolutional layer
    branch = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)

    # Combine paths
    output_tensor = Add()([main_path, branch])
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Adjust input feature dimensionality
    conv_initial = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_layer)

    # Level 1: Basic block
    level1_output = basic_block(conv_initial, filters=16)

    # Level 2: Two residual blocks
    level2_block1 = residual_block(level1_output, filters=32)
    level2_block2 = residual_block(level2_block1, filters=32)

    # Level 3: Global branch with convolutional layer
    global_branch_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(conv_initial)

    # Combine Level 2 and Level 3
    combined_output = Add()([level2_block2, global_branch_conv])

    # Pooling and fully connected layers
    avg_pool = AveragePooling2D(pool_size=(8, 8))(combined_output)
    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model