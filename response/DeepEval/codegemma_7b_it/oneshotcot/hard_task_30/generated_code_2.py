import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):

        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(input_tensor)
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(main_path)
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(main_path)

        branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(input_tensor)

        output_tensor = keras.layers.Add()([main_path, branch_path])

        return output_tensor

    block1_output = block1(input_tensor=input_layer)

    # Block 2
    def block2(input_tensor):

        input_tensor_split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_tensor)

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor_split[0])
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path1 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path1)

        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor_split[1])
        path2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor_split[2])
        path3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path3)

        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor

    block2_output = block2(input_tensor=block1_output)

    # Output layers
    bath_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model