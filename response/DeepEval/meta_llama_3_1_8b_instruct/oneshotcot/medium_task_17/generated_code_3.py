import keras
from keras.layers import Input, Reshape, Permute, Conv2D, Dense, Softmax

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    channels, height, width = input_layer.shape.as_list()[1:]
    groups = 3
    channels_per_group = channels // groups

    reshaped = Reshape((height, width, groups, channels_per_group))(input_layer)
    permutated = Permute((2, 3, 1, 4))(reshaped)
    reshaped_back = Reshape((height, width, channels))(permutated)

    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_back)

    flatten = keras.layers.Flatten()(conv)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model