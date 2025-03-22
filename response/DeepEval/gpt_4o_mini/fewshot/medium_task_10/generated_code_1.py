import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor):
    # Main path
    conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
    batch_norm = BatchNormalization()(conv)
    activation = ReLU()(batch_norm)
    
    # Branch path (identity connection)
    branch = input_tensor
    
    # Addition of main path and branch
    output_tensor = Add()([activation, branch])
    return output_tensor

def residual_block(input_tensor):
    # First main path
    main_path = basic_block(input_tensor)

    # Branch path with independent convolution
    branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_tensor)

    # Adding main path and branch
    output_tensor = Add()([main_path, branch])
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB

    # Level 1: Basic Block
    level1_output = basic_block(input_layer)
    
    # Level 2: Two Residual Blocks
    level2_output = residual_block(level1_output)
    level2_output = residual_block(level2_output)

    # Level 3: Global Branch
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(level1_output)
    
    # Final output from Level 3
    final_output = Add()([level2_output, global_branch])
    
    # Average pooling and fully connected layer
    pooled_output = AveragePooling2D(pool_size=(8, 8))(final_output)  # Global average pooling
    flatten = Flatten()(pooled_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # 10 classes for CIFAR-10

    # Construct model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model