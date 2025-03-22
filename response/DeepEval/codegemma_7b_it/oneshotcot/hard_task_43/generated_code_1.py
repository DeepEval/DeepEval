import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, AveragePooling2D

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    concat_block_1 = Concatenate()([path1, path2, path3])

    # Block 2
    branch_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_block_1)
    branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat_block_1)
    branch_3 = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(concat_block_1)
    branch_4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(concat_block_1)
    concat_block_2 = Concatenate()([branch_1, branch_2, branch_3, branch_4])
    
    # Fully connected layer
    flatten_block_2 = Flatten()(concat_block_2)
    dense_block_2 = Dense(units=256, activation='relu')(flatten_block_2)

    # Reshape block 1 output
    reshape_block_1 = Reshape((4, 4, 1))(concat_block_1)

    # Concatenate block 1 and block 2 outputs
    concat_final = Concatenate()([reshape_block_1, dense_block_2])

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(concat_final)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model