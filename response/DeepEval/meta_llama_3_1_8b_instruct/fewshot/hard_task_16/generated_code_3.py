import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, Dense, Reshape, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    transition_conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(block1_output)

    def block_2(input_tensor):
        global_max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        weights = Dense(units=16, activation='relu')(global_max_pool)
        weights = Dense(units=16, activation='relu')(weights)
        weights = Reshape(target_shape=(1, 1, 16))(weights)
        output_tensor = tf.multiply(input_tensor, weights)
        return output_tensor

    block2_output = block_2(input_tensor=transition_conv)

    branch_output = input_layer

    main_path_output = Add()([block2_output, branch_output])
    output_layer = Flatten()(main_path_output)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model