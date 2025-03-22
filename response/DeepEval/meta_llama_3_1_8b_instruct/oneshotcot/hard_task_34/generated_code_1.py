import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([input_tensor, conv])

        return output_tensor

    x = block(input_layer)
    x = block(x)
    x = block(x)

    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    fuse_path = Add()([x, branch_path])
    batch_norm = BatchNormalization()(fuse_path)
    flatten_layer = Flatten()(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model