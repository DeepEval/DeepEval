import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
        sep_conv = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(depthwise)
        add_output = Add()([input_tensor, sep_conv])
        return add_output

    branch1_output = block(input_layer)
    branch2_output = block(branch1_output)
    branch3_output = block(branch1_output)

    concat_layer = Concatenate(axis=-1)([branch1_output, branch2_output, branch3_output])
    batch_norm = BatchNormalization()(concat_layer)
    flatten = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    dense3 = Dense(units=32, activation='relu')(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()