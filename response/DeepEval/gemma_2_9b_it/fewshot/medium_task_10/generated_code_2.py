import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution to reduce dimensionality
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Basic block
    def basic_block(input_tensor):
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path = BatchNormalization()(main_path)
        branch_path = input_tensor
        output_tensor = Add()([main_path, branch_path])
        return output_tensor

    # Level 1
    x = basic_block(x)

    # Level 2 (Residual blocks)
    def residual_block(input_tensor):
        main_path = basic_block(input_tensor)
        branch_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Add()([main_path, branch_path])
        return output_tensor

    x = residual_block(x)
    x = residual_block(x)

    # Level 3 (Global branch)
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = Add()([x, global_branch]) 

    # Final processing
    x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid')(x)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model