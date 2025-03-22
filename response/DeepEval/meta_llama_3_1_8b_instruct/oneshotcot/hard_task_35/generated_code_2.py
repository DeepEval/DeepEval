import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # same block
    def same_block(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(8, 8), strides=(8, 8), padding='valid')(input_tensor)
        dense1 = Dense(units=128, activation='relu')(avg_pool)
        dense2 = Dense(units=64, activation='relu')(dense1)
        weights = Dense(units=64, activation='linear')(dense2)
        reshaped_weights = Reshape(target_shape=(1, 1, 64))(weights)
        element_wise_product = Multiply()([input_tensor, reshaped_weights])
        return element_wise_product
    
    branch1 = same_block(input_layer)
    branch2 = same_block(input_layer)
    
    # concatenate outputs from both branches
    concatenated_output = Concatenate()([branch1, branch2])
    
    # flatten and dense layer
    bath_norm = BatchNormalization()(concatenated_output)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model