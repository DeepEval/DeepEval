import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # First path
        conv_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_path = BatchNormalization()(conv_path)
        # Second path
        conv_path_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_path)
        conv_path_2 = BatchNormalization()(conv_path_2)
        # Concatenate the original input with the new features
        output_tensor = Concatenate(axis=-1)([input_tensor, conv_path_2])
        return output_tensor

    # Apply the block three times
    block_output = block(input_tensor=input_layer)
    block_output = block(input_tensor=block_output)
    block_output = block(input_tensor=block_output)

    # Second pathway
    input_layer_2 = Input(shape=(28, 28, 1))
    block_output_2 = block(input_tensor=input_layer_2)
    block_output_2 = block(input_tensor=block_output_2)
    block_output_2 = block(input_tensor=block_output_2)

    # Merge outputs from both pathways
    merged_output = Concatenate(axis=-1)([block_output, block_output_2])
    flatten_layer = Flatten()(merged_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=[input_layer, input_layer_2], outputs=output_layer)

    return model