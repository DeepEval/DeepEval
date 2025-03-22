import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(groups[1])
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def branch_path(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1

    main_path_output = main_path(input_tensor=input_layer)
    branch_path_output = branch_path(input_tensor=input_layer)
    concat_output = keras.layers.Add()([main_path_output, branch_path_output])
    flatten_output = keras.layers.Flatten()(concat_output)
    dense1_output = keras.layers.Dense(units=128, activation='relu')(flatten_output)
    dense2_output = keras.layers.Dense(units=10, activation='softmax')(dense1_output)

    model = keras.Model(inputs=input_layer, outputs=dense2_output)

    return model