import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def basic_block(input_tensor):
        path1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = BatchNormalization()(path1)
        path1 = LeakyReLU()(path1)
        path2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Add()([path1, path2])
        
        return output_tensor

    # Level 1
    basic_block_output = basic_block(conv)

    # Level 2
    level2_output = basic_block(basic_block_output)

    # Global Branch
    global_branch_output = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)

    # Level 3
    level3_output = Add()([global_branch_output, level2_output])

    # Average Pooling
    avg_pooling_output = AveragePooling2D(pool_size=(8, 8), strides=8, padding='valid')(level3_output)

    # Flatten
    flatten_output = Flatten()(avg_pooling_output)

    # Dense Layer
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model