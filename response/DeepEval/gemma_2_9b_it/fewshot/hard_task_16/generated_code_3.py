import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dense, Flatten, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs_groups[0])
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1_1)
        conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv1_2)
        conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs_groups[1])
        conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2_1)
        conv2_3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv2_2)
        conv3_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs_groups[2])
        conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv3_1)
        conv3_3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv3_2)
        output_tensor = Concatenate()([conv1_3, conv2_3, conv3_3])
        return output_tensor

    block1_output = block1(input_layer)

    transition_conv = Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(block1_output)

    def block2(input_tensor):
        pool = MaxPooling2D(pool_size=(7, 7))(transition_conv)
        dense1 = Dense(units=128, activation='relu')(pool)
        dense2 = Dense(units=transition_conv.shape[-1], activation='relu')(dense1)
        weights = Reshape(target_shape=(1, 1, transition_conv.shape[-1]))(dense2)
        output_tensor = tf.multiply(transition_conv, weights)
        return output_tensor

    block2_output = block2(transition_conv)

    branch_output = input_layer

    added_output = Add()([block2_output, branch_output])

    flatten = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model