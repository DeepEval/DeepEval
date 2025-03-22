import keras
from keras.layers import Input, Conv2D, Lambda, Reshape, Permute, concatenate, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    def block1(input_tensor):
        groups = 2
        input_shape = keras.backend.int_shape(input_tensor)
        shape_before_reshape = (input_shape[0], input_shape[1], input_shape[2] // groups, groups)
        x = Reshape(shape_before_reshape)(input_tensor)
        x = tf.split(x, num_or_size_splits=groups, axis=3)
        for i in range(groups):
            x[i] = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[i])
            x[i] = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[i])
            x[i] = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[i])
        x = concatenate(x, axis=3)
        x = Reshape(target_shape=input_shape)(x)
        return x

    block1_output = block1(conv1)

    # Block 2
    def block2(input_tensor):
        input_shape = keras.backend.int_shape(input_tensor)
        channels = input_shape[3]
        groups = 4
        shape_before_reshape = (input_shape[0], input_shape[1], input_shape[2] // groups, channels // groups)
        x = Reshape(shape_before_reshape)(input_tensor)
        x = Lambda(lambda z: tf.transpose(z, perm=[0, 1, 3, 2]))(x)
        x = tf.split(x, num_or_size_splits=groups, axis=-1)
        for i in range(groups):
            x[i] = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[i])
        x = concatenate(x, axis=3)
        x = Lambda(lambda z: tf.transpose(z, perm=[0, 1, 3, 2]))(x)
        x = Reshape(target_shape=input_shape)(x)
        return x

    block2_output = block2(block1_output)

    # Final layers
    flatten = keras.layers.Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model