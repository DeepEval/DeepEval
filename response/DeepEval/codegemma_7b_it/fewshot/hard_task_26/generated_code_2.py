import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
        branch2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = UpSampling2D(size=(2, 2))(branch2)
        branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
        branch3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch3 = UpSampling2D(size=(2, 2))(branch3)
        concat = Concatenate()([branch1, branch2, branch3])
        conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
        return conv2

    def branch_path(input_tensor):
        conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        return conv2

    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)
    added = keras.layers.Add()([main_path_output, branch_path_output])
    flatten = keras.layers.Flatten()(added)
    output_layer = keras.layers.Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model