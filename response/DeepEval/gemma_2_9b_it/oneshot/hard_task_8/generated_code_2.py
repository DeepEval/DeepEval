import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x_depthwise)

    branch_x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_x)

    x = Concatenate()([x, branch_x])

    # Block 2
    shape = keras.layers.Shape()(x)
    groups = 4
    x = Reshape((shape[1], shape[2], groups, shape[3] // groups))(x)
    x = Permute((1, 2, 4, 3))(x)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model