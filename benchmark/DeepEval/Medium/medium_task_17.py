from keras.models import Model
from keras.layers import Input, Reshape, Permute, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))     

    batchsize, height, width, channels = input_layer.shape
    groups = 3
    channels_per_group = channels // groups
    x = Reshape(target_shape=(height, width, groups, channels_per_group))(input_layer)
    x = Permute(dims=(1, 2, 4, 3))(x)
    x = Reshape(target_shape=(height, width, channels))(x)

    output_layer = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
