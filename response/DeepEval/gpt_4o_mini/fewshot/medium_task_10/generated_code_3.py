import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have a shape of 32x32x3

    # Initial convolution to adjust feature dimensionality to 16
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_layer)

    def basic_block(input_tensor):
        # Main path
        conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        
        # Branch
        branch = input_tensor
        
        # Combine both paths
        output_tensor = Add()([relu, branch])
        return output_tensor

    # Level 1: First basic block
    level_1_output = basic_block(initial_conv)

    # Level 2: Two residual blocks
    def residual_block(input_tensor):
        main_path = basic_block(input_tensor)
        branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_tensor)
        output_tensor = Add()([main_path, branch])
        return output_tensor

    level_2_output = residual_block(level_1_output)
    level_2_output = residual_block(level_2_output)

    # Level 3: Capture features with a convolution layer in the global branch
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(initial_conv)
    output_tensor = Add()([level_2_output, global_branch])

    # Final layers: Average pooling and a fully connected layer
    avg_pooling = AveragePooling2D(pool_size=(8, 8))(output_tensor)  # CIFAR-10 size is 32x32, so pool by 8
    flatten_layer = Flatten()(avg_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    model = Model(inputs=input_layer, outputs=output_layer)

    return model