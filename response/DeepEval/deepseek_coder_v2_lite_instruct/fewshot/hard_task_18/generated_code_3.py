import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Multiply, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool

    def block_2(input_tensor):
        # Main path
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=32, activation='relu')(avg_pool)  # Assuming the number of channels in the output of block 1 is 32
        dense2 = Dense(units=32, activation='relu')(dense1)
        reshaped_weights = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense2)  # Reshape to match the input tensor shape
        
        # Multiply weights back to the input
        weights_multiplied = Multiply()([input_tensor, reshaped_weights])
        
        # Flatten the result
        flatten_layer = Flatten()(weights_multiplied)
        
        # Final classification layer
        output_layer = Dense(units=10, activation='softmax')(flatten_layer)
        
        return output_layer

    # First block
    first_block_output = block_1(input_layer)
    
    # Second block
    final_output = block_2(first_block_output)

    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model