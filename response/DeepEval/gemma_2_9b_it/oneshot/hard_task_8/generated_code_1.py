import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, concatenate, Reshape, Permute, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x_depthwise)

    branch_x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_x)
    x = concatenate([x, branch_x], axis=3)  

    # Block 2
    shape = keras.backend.shape(x)
    x = Reshape((shape[1], shape[2], 4, shape[3] // 4))(x)
    x = Permute((0, 1, 3, 2))(x)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model