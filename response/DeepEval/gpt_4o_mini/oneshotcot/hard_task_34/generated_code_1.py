import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, ReLU, BatchNormalization, SeparableConv2D, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def main_path_block(input_tensor):
        # Apply separable convolution and ReLU activation
        conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        activation = ReLU()(conv)
        # Batch normalization after activation
        batch_norm = BatchNormalization()(activation)
        return batch_norm

    # Main path
    main_path = main_path_block(input_layer)
    main_path = main_path_block(main_path)
    main_path = main_path_block(main_path)

    # Branch path with convolution layer maintaining the same number of channels
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(main_path)

    # Fuse the main path and branch path
    fused = Add()([main_path, branch_path])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(fused)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model