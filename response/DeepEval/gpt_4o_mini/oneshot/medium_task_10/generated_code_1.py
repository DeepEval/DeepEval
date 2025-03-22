import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense

def basic_block(input_tensor):
    # Main path
    conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)
    
    # Branch path (identity connection)
    branch = input_tensor
    
    # Combine main and branch paths
    output_tensor = Add()([relu, branch])
    
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # First level: one basic block
    level1_output = basic_block(input_layer)

    # Second level: two residual blocks
    level2_output = basic_block(level1_output)
    level2_output = basic_block(level2_output)

    # Global branch
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(level1_output)

    # Combine second level output with global branch
    final_output = Add()([level2_output, global_branch])

    # Average pooling followed by fully connected layer
    pooled_output = GlobalAveragePooling2D()(final_output)
    dense_output = Dense(units=10, activation='softmax')(pooled_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense_output)

    return model