import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense

def basic_block(input_tensor):
    # Main path
    main_path = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
    main_path = BatchNormalization()(main_path)
    main_path = ReLU()(main_path)

    # Branch
    branch_path = input_tensor

    # Combine the paths
    output_tensor = Add()([main_path, branch_path])
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 input shape

    # First level
    level1_output = basic_block(input_layer)

    # Second level (Two Residual Blocks)
    level2_output = level1_output
    for _ in range(2):
        level2_output = basic_block(level2_output)

    # Third level (Adding output from the second level to a convolutional layer)
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_layer)
    level3_output = Add()([level2_output, global_branch])

    # Global Average Pooling and Fully Connected Layer
    pooling_layer = GlobalAveragePooling2D()(level3_output)
    output_layer = Dense(units=10, activation='softmax')(pooling_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model