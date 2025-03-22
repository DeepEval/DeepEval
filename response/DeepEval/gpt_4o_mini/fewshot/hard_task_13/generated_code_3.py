import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    def block_1(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)

        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)

    # Second Block
    global_avg_pool = GlobalAveragePooling2D()(block1_output)
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=block1_output.shape[-1])(dense1)  # Same number of units as the channel size of block 1 output
    reshaped_weights = Reshape(target_shape=(1, 1, block1_output.shape[-1]))(dense2)
    
    weighted_features = Multiply()([block1_output, reshaped_weights])
    
    # Final output layer
    flatten = GlobalAveragePooling2D()(weighted_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model