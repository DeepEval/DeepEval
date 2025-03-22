import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        branch3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv1)
        branch4 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch5 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch4)
        branch6 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch5)
        branch7 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(branch6)
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6, branch7])
        return output_tensor

    def branch_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(conv1)
        return output_tensor

    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)
    added_output = Lambda(lambda x: tf.add(x[0], x[1]))([main_path_output, branch_path_output])
    flattened_output = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flattened_output)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model