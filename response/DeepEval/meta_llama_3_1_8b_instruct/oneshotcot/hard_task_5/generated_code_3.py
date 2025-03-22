import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute, Lambda, Add
from keras import backend as K
from tensorflow.keras import layers

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        group_size = 3
        channels = K.int_shape(input_tensor)[-1]
        split_input = Lambda(lambda x: K.split(x, group_size, axis=-1))(input_tensor)
        path1 = Conv2D(filters=channels // group_size, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[0])
        path2 = Conv2D(filters=channels // group_size, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[1])
        path3 = Conv2D(filters=channels // group_size, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[2])
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        shape = K.int_shape(input_tensor)
        group_size = 3
        channels_per_group = shape[-1] // group_size
        input_shape = (shape[1], shape[2], group_size, channels_per_group)
        reshaped = Reshape(input_shape)(input_tensor)
        permuted = Permute((1, 3, 2, 4))(reshaped)
        output_tensor = Reshape(shape)(permuted)
        return output_tensor
    
    block2_output = block2(block1_output)

    # Block 3
    block3_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise=True)(block2_output)

    # Branch
    branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine outputs
    combined = Add()([block3_output, branch])

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(Flatten()(combined))

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model