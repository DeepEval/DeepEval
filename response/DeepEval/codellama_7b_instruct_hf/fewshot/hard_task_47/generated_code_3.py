import keras
from keras.layers import Input, Lambda, Dense, Conv2D, DepthwiseConv2D, Flatten, Concatenate, BatchNormalization, Reshape
from keras.applications import VGG16


def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    def block_1(input_tensor):
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        branch_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch_3 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch_4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        output_tensor = Concatenate()([branch_1, branch_2, branch_3, branch_4])
        return output_tensor

    output_tensor = block_1(inputs_groups[0])
    bn = BatchNormalization()(output_tensor)
    output_tensor = block_2(bn)
    fc1 = Dense(units=128, activation='relu')(output_tensor)
    fc2 = Dense(units=10, activation='softmax')(fc1)

    model = keras.Model(inputs=input_layer, outputs=fc2)

    return model