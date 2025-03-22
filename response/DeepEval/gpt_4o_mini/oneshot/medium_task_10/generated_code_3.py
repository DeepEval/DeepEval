import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense

def basic_block(input_tensor):
    # Main path
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Branch (skip connection)
    branch = input_tensor

    # Combine paths
    output_tensor = Add()([x, branch])
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB
    
    # First level: basic block
    level1_output = basic_block(input_layer)
    
    # Second level: two residual blocks
    level2_output = basic_block(level1_output)
    level2_output = basic_block(level2_output)

    # Adding an independent convolutional layer in the branch for the second level
    branch_conv = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(level1_output)
    level2_output = Add()([level2_output, branch_conv])

    # Third level: using a convolutional layer in the global branch
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(level1_output)
    
    # Final combination of second level output with global branch
    level3_output = Add()([level2_output, global_branch])

    # Global average pooling and dense layer for classification
    pooled_output = GlobalAveragePooling2D()(level3_output)
    output_layer = Dense(units=10, activation='softmax')(pooled_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model