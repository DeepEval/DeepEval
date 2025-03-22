import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def basic_block(input_tensor):
        # Main path
        conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
        bn = BatchNormalization()(conv)
        relu = ReLU()(bn)

        # Branch path
        branch = input_tensor

        # Feature fusion
        output_tensor = Add()([relu, branch])
        return output_tensor

    # Initial convolutional layer to reduce dimensionality
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_layer)

    # First basic block
    block1_output = basic_block(initial_conv)

    # Second basic block
    block2_output = basic_block(block1_output)

    # Additional convolutional layer in the branch
    branch_conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(initial_conv)

    # Feature fusion between block output and branch output
    final_output = Add()([block2_output, branch_conv])

    # Average pooling and flattening
    pooled_output = AveragePooling2D(pool_size=(2, 2))(final_output)
    flatten_layer = Flatten()(pooled_output)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model