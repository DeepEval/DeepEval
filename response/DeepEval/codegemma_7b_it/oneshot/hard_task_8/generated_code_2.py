import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D, Permute

def dl_model(): 
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    def block2(input_tensor):
        conv1 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        return conv2

    block1_output = block1(input_tensor=input_layer)
    block2_output = block2(input_tensor=input_layer)

    concat_output = Concatenate()([block1_output, block2_output])

    # Block 2
    shape_output = keras.backend.int_shape(concat_output)
    reshape_output = Reshape((shape_output[1], shape_output[2], shape_output[3], 2))(concat_output)
    permute_output = Permute((1, 2, 4, 3))(reshape_output)
    reshape_output2 = Reshape((shape_output[1], shape_output[2], shape_output[3]*2))(permute_output)

    # Fully connected layer
    flatten_layer = Flatten()(reshape_output2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model