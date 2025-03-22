import keras
from keras.layers import Input, ReLU, SeparableConv2D, MaxPooling2D, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def main_block(input_tensor):
        x = ReLU()(input_tensor)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
        return x
    
    main_path_1 = main_block(input_layer)
    main_path_2 = main_block(main_path_1)
    
    # Branch path
    branch_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Ensure both paths have the same dimensions
    branch_path = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(branch_conv)
    
    # Add the outputs from both paths
    added_paths = Add()([main_path_2, branch_path])
    
    # Final layers
    flatten_layer = Flatten()(added_paths)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model