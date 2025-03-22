import keras
from keras.layers import Input, Reshape, Permute, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    shape_as_list = keras.backend.int_shape(input_layer)
    height, width, channels = shape_as_list[1], shape_as_list[2], shape_as_list[3]

    x = Reshape((height, width, 3, channels // 3))(input_layer)
    x = Permute((0, 1, 3, 2))(x)
    x = Reshape((height, width, channels))(x)

    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model