import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    primary_path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(primary_path)
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(primary_path)

    branch_path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)

    concat_path = Concatenate()([primary_path, branch_path])

    # Block 2
    block2_shape = keras.backend.int_shape(concat_path)
    reshaped = Reshape(target_shape=(block2_shape[1], block2_shape[2], block2_shape[3] * block2_shape[4]))(concat_path)

    groups = 4
    reshaped = Permute((2, 1, 3))(reshaped)
    reshaped = Reshape((block2_shape[1], block2_shape[2], groups, block2_shape[3] // groups))(reshaped)

    # Channel shuffling
    reshaped = Permute((3, 1, 2, 4))(reshaped)

    reshaped = Reshape((block2_shape[1], block2_shape[2], block2_shape[3] * block2_shape[4]))(reshaped)

    # Final fully connected layer
    output_layer = Dense(units=10, activation='softmax')(reshaped)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model