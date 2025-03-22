import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block(input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        activation = ReLU()(batch_norm)
        return activation

    # First block
    block1_output = block(input_tensor=input_layer, filters=32)
    block1_output = block(input_tensor=block1_output, filters=32)
    path1 = block1_output

    # Second block
    block2_output = block(input_tensor=input_layer, filters=64)
    block2_output = block(input_tensor=block2_output, filters=64)
    path2 = block2_output

    # Third block
    block3_output = block(input_tensor=input_layer, filters=128)
    block3_output = block(input_tensor=block3_output, filters=128)
    path3 = block3_output

    # Parallel branch
    parallel_branch = block(input_tensor=input_layer, filters=128)
    parallel_branch = block(input_tensor=parallel_branch, filters=128)

    # Add all paths together
    added_output = Add()([path1, path2, path3, parallel_branch])

    # Flatten the result
    flattened_output = Flatten()(added_output)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model