from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def se_block(input_tensor):
        channels = input_tensor.shape[-1]
        x = GlobalAveragePooling2D()(input_tensor)
        x = Dense(channels // 16, activation='relu')(x)
        x = Dense(channels, activation='sigmoid')(x)
        x = Reshape((1, 1, channels))(x)
        scaled_tensor = Multiply()([input_tensor, x])
        return scaled_tensor

    branch1 = se_block(input_layer)
    branch2 = se_block(input_layer)

    concatenated_branches = Concatenate()([branch1, branch2])
    flatten_layer = Flatten()(concatenated_branches)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model