import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense

def dl_model():  
    input_layer = Input(shape=(28, 28, 1))

    def residual_block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        return Add()([input_tensor, x])

    branch1 = residual_block(input_layer)
    branch2 = residual_block(branch1)
    branch3 = residual_block(branch2)

    concatenated_output = Concatenate()([branch1, branch2, branch3])
    flatten_layer = Flatten()(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model