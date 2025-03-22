import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor, filters):
    # Main path: Conv -> BatchNorm -> ReLU
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Branch: Direct connection to input
    return Add()([x, input_tensor])

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution to adjust dimensionality to 16
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_layer)

    # Level 1: Basic block
    level1_output = basic_block(x, filters=16)

    # Level 2: Two residual blocks
    # First residual block
    main_path1 = basic_block(level1_output, filters=32)
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(level1_output)
    level2_output1 = Add()([main_path1, branch1])

    # Second residual block
    main_path2 = basic_block(level2_output1, filters=32)
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(level2_output1)
    level2_output2 = Add()([main_path2, branch2])

    # Level 3: Global branch with convolution
    global_branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x)

    # Addition of global branch to the last output of level 2
    level3_output = Add()([global_branch, level2_output2])

    # Average pooling and fully connected layer
    pooled = AveragePooling2D(pool_size=(8, 8))(level3_output)
    flatten_layer = Flatten()(pooled)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Define and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model