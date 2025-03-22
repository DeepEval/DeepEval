import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Concatenate, Add, Flatten, Dense, Activation

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    def specific_block(input_tensor):
        relu_activation = Activation('relu')(input_tensor)
        sep_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu_activation)
        output_tensor = Concatenate(axis=-1)([input_tensor, sep_conv])
        return output_tensor

    # Main path with specific block repeated three times
    main_path = specific_block(input_layer)
    main_path = specific_block(main_path)
    main_path = specific_block(main_path)

    # Branch path
    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse features from both paths
    fused_features = Add()([main_path, branch_path])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model