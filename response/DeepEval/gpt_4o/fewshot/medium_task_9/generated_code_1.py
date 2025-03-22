import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, Activation

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution to reduce dimensionality
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    def basic_block(input_tensor):
        # Main path
        conv_main = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        bn_main = BatchNormalization()(conv_main)
        relu_main = Activation('relu')(bn_main)
        
        # Branch path - direct connection
        branch_path = input_tensor

        # Feature fusion
        output_tensor = Add()([relu_main, branch_path])
        return output_tensor

    # Main structure with two consecutive basic blocks
    block1_output = basic_block(input_tensor=initial_conv)
    block2_output = basic_block(input_tensor=block1_output)

    # Branch path - feature extraction
    conv_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Feature fusion from the main structure and branch path
    fusion_output = Add()([block2_output, conv_branch])

    # Downsampling
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(fusion_output)
    
    # Classification head
    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model