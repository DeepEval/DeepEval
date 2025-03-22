import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer to reduce dimensionality to 16
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the basic block
    def basic_block(input_tensor):
        # Main path
        main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path = BatchNormalization()(main_path)
        # Branch path
        branch_path = input_tensor
        # Feature fusion by adding the outputs from both paths
        adding_layer = Add()([main_path, branch_path])
        return adding_layer

    # Apply two consecutive basic blocks
    block1 = basic_block(conv)
    block2 = basic_block(block1)

    # Define the branch for feature extraction
    branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs from both paths
    adding_layer = Add()([block2, branch])

    # Apply average pooling to downsample the feature map
    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='same')(adding_layer)

    # Flatten the feature map
    flatten_layer = Flatten()(avg_pool)

    # Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model