import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Multiply, Concatenate, Dense, Activation, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block_1(input_tensor):
        # Path 1: Global Average Pooling followed by two fully connected layers
        gap1 = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=64, activation='relu')(gap1)
        dense2 = Dense(units=32, activation='relu')(dense1)
        path1_output = dense2

        # Path 2: Global Max Pooling followed by two fully connected layers
        gmp1 = GlobalMaxPooling2D()(input_tensor)
        dense1_gmp = Dense(units=64, activation='relu')(gmp1)
        dense2_gmp = Dense(units=32, activation='relu')(dense1_gmp)
        path2_output = dense2_gmp

        # Add the outputs of both paths
        added_output = Add()([path1_output, path2_output])

        # Generate channel attention weights
        reshaped = Reshape((1, 1, 64))(added_output)  # Adjust dimensions to match input
        activation_output = Activation('sigmoid')(reshaped)

        # Apply channel attention weights to the original features
        channel_attention_output = Multiply()([input_tensor, activation_output])

        return channel_attention_output

    def block_2(input_tensor):
        # Separate average and max pooling
        avg_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(input_tensor)
        max_pool = GlobalMaxPooling2D()(input_tensor)

        # Concatenate along the channel dimension
        concatenated = Concatenate(axis=-1)([avg_pool, max_pool])

        # 1x1 convolution to further refine features
        conv1x1 = Conv2D(filters=3, kernel_size=(1, 1), activation='sigmoid')(concatenated)

        return conv1x1

    # Apply Block 1
    block1_output = block_1(input_tensor=input_layer)

    # Apply Block 2
    block2_output = block_2(input_tensor=block1_output)

    # Additional branch to ensure output channels align with input channels
    conv_branch = Conv2D(filters=3, kernel_size=(1, 1), activation='linear')(block2_output)

    # Add the main path and the additional branch
    final_output = Add()([block1_output, conv_branch])

    # Flatten and pass through a fully connected layer for classification
    flattened = Flatten()(final_output)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model