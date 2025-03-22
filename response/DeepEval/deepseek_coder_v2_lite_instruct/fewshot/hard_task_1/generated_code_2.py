import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Multiply, Concatenate, Dense, Activation, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Block 1: Channel-wise feature extraction
    def block_1(input_tensor):
        # Path 1: Global Average Pooling followed by two fully connected layers
        gap = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=64, activation='relu')(gap)
        dense2 = Dense(units=32, activation='relu')(dense1)
        path1_output = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense2)

        # Path 2: Global Max Pooling followed by two fully connected layers
        gmp = GlobalMaxPooling2D()(input_tensor)
        dense1_gmp = Dense(units=64, activation='relu')(gmp)
        dense2_gmp = Dense(units=32, activation='relu')(dense1_gmp)
        path2_output = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense2_gmp)

        # Add the outputs and apply element-wise multiplication to adjust channel attention
        added_output = Add()([path1_output, path2_output])
        attention_weights = Activation('sigmoid')(added_output)
        adjusted_input = Multiply()([input_tensor, attention_weights])

        return adjusted_input

    # Apply Block 1 to the input
    block1_output = block_1(input_layer)

    # Block 2: Spatial feature extraction
    def block_2(input_tensor):
        # Separate paths for Average and Max Pooling
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2)(input_tensor)
        max_pool = GlobalMaxPooling2D()(input_tensor)

        # Concatenate the outputs along the channel dimension
        concatenated = Concatenate(axis=-1)([avg_pool, max_pool])

        # 1x1 convolution to normalize features
        conv1x1 = Conv2D(filters=3, kernel_size=(1, 1), activation='sigmoid')(concatenated)

        # Reshape to ensure channels match
        reshaped = Reshape(target_shape=(32, 32, 3))(conv1x1)

        return reshaped

    # Apply Block 2 to the output of Block 1
    block2_output = block_2(block1_output)

    # Additional branch to ensure output channels align with input channels
    conv_branch = Conv2D(filters=3, kernel_size=(1, 1), activation='linear')(block1_output)

    # Add and activate the final output
    final_output = Add()([block2_output, conv_branch])
    flattened = Flatten()(final_output)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage
model = dl_model()
model.summary()