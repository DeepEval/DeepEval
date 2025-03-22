import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Main path
        conv_main = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Branch path
        conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Concatenate along the channel dimension
        output_tensor = Concatenate(axis=-1)([conv_main, conv_branch])
        return output_tensor

    # Apply the block three times in sequence
    block_output = block(input_tensor=input_layer)
    block_output = block(input_tensor=block_output)
    block_output = block(input_tensor=block_output)

    # Addition operation to fuse features from both paths
    fused_features = keras.layers.add([block_output, block_output])

    # Flatten the result
    flatten_layer = Flatten()(fused_features)

    # Pass through a fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model