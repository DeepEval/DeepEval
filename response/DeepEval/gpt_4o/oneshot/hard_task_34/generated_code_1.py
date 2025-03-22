import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Add, Flatten, Dense, ReLU, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def main_path_block(input_tensor):
        # Feature extraction and enhancement block
        relu = ReLU()(input_tensor)
        sep_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        concat = Concatenate()([input_tensor, sep_conv])
        return concat

    x = input_layer
    # Repeat the block three times
    for _ in range(3):
        x = main_path_block(x)

    main_path_output = x

    # Branch path
    branch_conv = Conv2D(filters=main_path_output.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    # Fuse features through addition
    fused_output = Add()([main_path_output, branch_conv])
    
    # Flatten and output layer
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model