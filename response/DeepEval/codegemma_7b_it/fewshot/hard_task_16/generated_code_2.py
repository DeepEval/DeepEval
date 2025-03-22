import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Reshape, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def transition_conv(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    def block_2(input_tensor):
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        fc_1 = Dense(units=32 * 8 * 8, activation='relu')(maxpool)
        fc_2 = Dense(units=32 * 8 * 8, activation='relu')(fc_1)
        weights = Reshape((8, 8, 32))(fc_2)
        main_path = tf.matmul(tf.transpose(maxpool, perm=[0, 1, 3, 2]), weights)
        branch = transition_conv(input_tensor)
        output_tensor = tf.add(main_path, branch)
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    transition_output = transition_conv(input_tensor=block1_output)
    block2_output = block_2(input_tensor=transition_output)
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model