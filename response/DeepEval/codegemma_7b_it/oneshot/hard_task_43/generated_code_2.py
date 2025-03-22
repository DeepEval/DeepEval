import keras
from keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block1(input_tensor):
        avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_tensor)

        flatten_1x1 = Flatten()(avg_pool_1x1)
        flatten_2x2 = Flatten()(avg_pool_2x2)
        flatten_4x4 = Flatten()(avg_pool_4x4)

        output_tensor = Concatenate()([flatten_1x1, flatten_2x2, flatten_4x4])

        return output_tensor

    def block2(input_tensor):
        conv_branch_1x1_3x3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_branch_1x1_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch_1x1_3x3)

        conv_branch_1x1_1x7_7x1_3x3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_branch_1x1_1x7_7x1_3x3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv_branch_1x1_1x7_7x1_3x3)
        conv_branch_1x1_1x7_7x1_3x3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv_branch_1x1_1x7_7x1_3x3)
        conv_branch_1x1_1x7_7x1_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch_1x1_1x7_7x1_3x3)

        avg_pool = AveragePooling2D(pool_size=(3, 3), strides=3, padding='valid')(input_tensor)

        output_tensor = Concatenate()([conv_branch_1x1_3x3, conv_branch_1x1_1x7_7x1_3x3, avg_pool])

        return output_tensor

    block1_output = block1(input_tensor)
    reshape_layer = Reshape((block1_output.shape[1], block1_output.shape[2], 1))(block1_output)
    dense_layer = Dense(units=128, activation='relu')(reshape_layer)
    block2_output = block2(dense_layer)

    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model