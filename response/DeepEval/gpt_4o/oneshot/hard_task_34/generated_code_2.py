import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Add, Flatten, Dense, Activation, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def main_path_block(input_tensor):
        # Apply ReLU activation
        relu_activated = Activation('relu')(input_tensor)
        # Apply Separable Convolution
        separable_conv = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu_activated)
        # Concatenate input with convolved feature map
        concatenated = Concatenate(axis=-1)([input_tensor, separable_conv])
        return concatenated

    # Main path: Repeat the block three times
    main_path = main_path_block(input_layer)
    main_path = main_path_block(main_path)
    main_path = main_path_block(main_path)
    
    # Branch path: Convolution to maintain the same channels as main_path output
    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    # Fuse the features using addition operation
    fused_features = Add()([main_path, branch_path])

    # Flatten and pass through a fully connected layer for classification
    flatten_layer = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model