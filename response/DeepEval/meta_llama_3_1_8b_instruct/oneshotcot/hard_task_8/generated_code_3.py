import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, Reshape, Permute, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    block1_path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_path2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_path2)
    block1_path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_path2)
    block1_branch = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_branch)
    block1_output = Concatenate()([block1_path1, block1_path3, block1_branch])

    # Block 2
    block2_output = block1_output
    block2_output = Reshape((14, 14, 3, 64))(block2_output)
    block2_output = Permute((1, 2, 4, 3))(block2_output)
    block2_output = Reshape((14, 14, 64))(block2_output)

    bath_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model