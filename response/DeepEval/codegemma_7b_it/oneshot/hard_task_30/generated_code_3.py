import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, DepthwiseConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def residual_block(input_tensor, filters):
        main_path = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch_path = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = keras.layers.add([main_path, branch_path])
        return output_tensor

    block1 = residual_block(input_tensor, filters=32)
    block1 = residual_block(block1, filters=32)

    block2 = residual_block(block1, filters=64)
    block2 = residual_block(block2, filters=64)

    block3 = residual_block(block2, filters=128)
    block3 = residual_block(block3, filters=128)

    block4 = Lambda(lambda x: tf.split(x, 3, axis=-1))(block3)
    block4 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block4[0])
    block4 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block4[1])
    block4 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block4[2])

    block4 = Lambda(lambda x: tf.concat(x, axis=-1))(block4)

    block5 = residual_block(block4, filters=256)
    block5 = residual_block(block5, filters=256)

    block6 = residual_block(block5, filters=512)
    block6 = residual_block(block6, filters=512)

    block7 = residual_block(block6, filters=1024)
    block7 = residual_block(block7, filters=1024)

    flatten_layer = Flatten()(block7)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model