import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def block1(input_tensor):
        x1 = Lambda(lambda x: tf.split(x, 3, axis=3))(input_tensor)
        x2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(x1[0])
        x3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(x1[1])
        x4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(x1[2])
        output_tensor = Concatenate()([x2, x3, x4])
        return output_tensor

    def block2(input_tensor):
        shape_before_reshape = keras.backend.int_shape(input_tensor)
        x = Reshape((shape_before_reshape[1], shape_before_reshape[2], 3, 16))(input_tensor)
        x = Permute((0, 1, 3, 2))(x)
        x = Reshape((shape_before_reshape[1], shape_before_reshape[2], 16))(x)
        return x

    def block3(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal')(input_tensor)
        return x

    # Branch path
    branch_output = MaxPooling2D(pool_size=(32, 32), strides=(32, 32), padding='valid')(input_layer)

    # Concatenation
    main_output = block1(input_layer)
    main_output = block2(main_output)
    main_output = block3(main_output)

    concat_output = Concatenate()([main_output, branch_output])

    # Classification layer
    flatten_layer = Flatten()(concat_output)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model