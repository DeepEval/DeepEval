import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Initial convolutional layer to adjust feature dimensionality
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Define the basic block
    def basic_block(input_tensor):
        main_path = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = ReLU()(main_path)
        
        # The branch connects directly to the input
        branch_path = input_tensor
        
        # Combine both paths
        output_tensor = Add()([main_path, branch_path])
        return output_tensor

    # Level 1: Basic block
    level_1_output = basic_block(initial_conv)

    # Level 2: Two residual blocks
    def residual_block(input_tensor):
        main_path = basic_block(input_tensor)
        branch_path = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_tensor)
        output_tensor = Add()([main_path, branch_path])
        return output_tensor

    level_2_output = residual_block(level_1_output)
    level_2_output = residual_block(level_2_output)

    # Level 3: Global branch with convolution and addition
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(initial_conv)
    combined_output = Add()([global_branch, level_2_output])

    # Final layers
    avg_pool = AveragePooling2D(pool_size=(8, 8))(combined_output)  # Pooling to reduce spatial dimensions
    flatten_layer = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model