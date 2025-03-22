import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras import backend as K

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def block_1(input_tensor):
        x1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([x1, x2, x3])
        return output_tensor

    def transition_conv(input_tensor):
        x = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return x

    def block_2(input_tensor):
        input_channels = input_tensor.shape[-1]
        x = GlobalMaxPooling2D()(input_tensor)
        y = Dense(units=input_channels, activation='relu')(x)
        y = Dense(units=input_channels, activation='sigmoid')(y)
        y = Lambda(lambda z: K.reshape(z, (-1, 1, 1, input_channels)))(y)
        out = Lambda(lambda x, y: x * y)([input_tensor, y])
        return out

    block1_output = block_1(input_tensor)
    branch1 = Lambda(lambda x: K.expand_dims(x, axis=1))(block1_output)
    branch2 = block_2(block1_output)
    main_output = Add()([branch1, branch2])
    main_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(main_output)
    main_output = Flatten()(main_output)
    output_layer = Dense(units=10, activation='softmax')(main_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model