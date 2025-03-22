import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal',
                       depthwise_constraint=keras.constraints.DepthwiseConstraint(min_depth=1, max_depth=64))(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor

    block1 = block(input_tensor)
    block2 = block(block1)
    block3 = block(block2)

    concat = Concatenate()([block1, block2, block3])
    flatten_layer = Flatten()(concat)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model