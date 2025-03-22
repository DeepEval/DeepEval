import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Multiply, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    def block1(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block1_output = block1(input_tensor=input_layer)

    # Second Block
    def block2(input_tensor):
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        
        # Fully connected layers for generating channel weights
        dense1 = Dense(units=global_avg_pool.shape[-1], activation='relu')(global_avg_pool)
        dense2 = Dense(units=global_avg_pool.shape[-1], activation='sigmoid')(dense1)
        
        # Reshape to match the input tensor's shape for channel-wise multiplication
        reshaped_weights = Reshape((1, 1, global_avg_pool.shape[-1]))(dense2)
        
        # Element-wise multiplication
        scaled_features = Multiply()([input_tensor, reshaped_weights])

        return scaled_features

    block2_output = block2(input_tensor=block1_output)

    # Final fully connected layer for classification
    flatten_layer = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model