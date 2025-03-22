import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32,32,3))

    def group_input(input_tensor, group_size):
        shape_input = tf.keras.backend.int_shape(input_tensor)
        batch_size, height, width, channels = shape_input
        input_reshaped = tf.reshape(input_tensor, shape=(batch_size, height, width, group_size, channels//group_size))
        input_transposed = tf.transpose(input_reshaped, perm=[0, 1, 2, 4, 3])
        return tf.reshape(input_transposed, shape=[batch_size, height, width, channels])

    group1 = group_input(input_tensor=input_layer, group_size=3)
    group2 = group_input(input_tensor=input_layer, group_size=4)
    group3 = group_input(input_tensor=input_layer, group_size=5)

    concat = keras.layers.concatenate([group1, group2, group3])

    swap_axis = Permute((2, 3, 1))(concat)

    reshape_output = Reshape((32, 32, 3))(swap_axis)

    flatten = Flatten()(reshape_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model