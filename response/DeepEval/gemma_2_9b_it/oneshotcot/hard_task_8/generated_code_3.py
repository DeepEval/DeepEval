import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_primary = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x_primary = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x_primary)

    x_branch = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x_branch)
    x = Concatenate()([x_primary, x_branch])

    # Block 2
    shape = keras.backend.shape(x)
    x = Reshape(target_shape=(shape[1], shape[2], 4, shape[3] // 4))(x)
    x = Permute(axes=(0, 1, 3, 2))(x)
    x = Reshape(target_shape=(shape[1], shape[2], shape[3]))(x)

    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model