import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    avg_pool_1x1_flatten = Flatten()(avg_pool_1x1)
    avg_pool_2x2_flatten = Flatten()(avg_pool_2x2)
    avg_pool_4x4_flatten = Flatten()(avg_pool_4x4)
    
    block_1_concat = Concatenate()([avg_pool_1x1_flatten, avg_pool_2x2_flatten, avg_pool_4x4_flatten])
    
    # Block 2
    block_2_input = Dense(units=128, activation='relu')(block_1_concat)
    block_2_reshape = Reshape(target_shape=(1, 1, 128))(block_2_input)
    
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block_2_reshape)
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_2_reshape)
    
    conv_1x7_7x1 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(block_2_reshape)
    conv_7x1_1x7 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(block_2_reshape)
    conv_3x3_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_2_reshape)
    
    avg_pool = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(block_2_reshape)
    
    block_2_concat = Concatenate()([conv_1x1, conv_3x3, conv_1x7_7x1, conv_7x1_1x7, conv_3x3_2, avg_pool])
    
    # Output layers
    flatten = Flatten()(block_2_concat)
    dense1 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model