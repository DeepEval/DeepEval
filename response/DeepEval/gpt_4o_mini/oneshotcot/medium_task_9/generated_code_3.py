import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor):
    # Main path
    main_path = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
    main_path = BatchNormalization()(main_path)
    main_path = ReLU()(main_path)

    # Branch path (skip connection)
    branch_path = input_tensor

    # Feature fusion by adding both paths
    output_tensor = Add()([main_path, branch_path])
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels (RGB)

    # Initial convolutional layer
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_layer)

    # First basic block
    block1_output = basic_block(initial_conv)

    # Second basic block
    block2_output = basic_block(block1_output)

    # Branch to extract additional features
    branch_features = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(block1_output)

    # Combine outputs again through addition
    combined_output = Add()([block2_output, branch_features])

    # Average pooling layer
    avg_pool = AveragePooling2D(pool_size=(2, 2))(combined_output)

    # Flatten the result
    flatten_layer = Flatten()(avg_pool)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model