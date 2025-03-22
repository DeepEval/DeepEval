import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, GlobalAveragePooling2D, Dense

def basic_block(input_tensor, filters=16):
    """A basic block consisting of a convolutional layer, batch normalization, and ReLU activation."""
    # Main path
    main_path = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
    main_path = BatchNormalization()(main_path)
    main_path = keras.activations.relu(main_path)

    # Branch directly connected to input
    branch = input_tensor
    
    # Add the main path and branch
    output_tensor = Add()([main_path, branch])

    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 channels (RGB)

    # First level
    level1 = basic_block(input_layer)

    # Second level with two residual blocks
    level2_path1 = basic_block(level1)
    level2_path2 = basic_block(level2_path1)
    
    # Second block with independent convolution for the branch
    branch_level2 = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(level1)
    
    # Add outputs of the main path and the branch
    level2_output = Add()([level2_path2, branch_level2])

    # Third level with a convolutional layer in the global branch
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(level1)
    
    # Final addition
    final_output = Add()([level2_output, global_branch])

    # Global average pooling and output layer
    pooled_output = GlobalAveragePooling2D()(final_output)
    output_layer = Dense(units=10, activation='softmax')(pooled_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model