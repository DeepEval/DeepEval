import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
        return max_pooling

    def branch_path(input_tensor):
        gap = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(gap)
        dense2 = Dense(units=10, activation='relu')(dense1)
        reshape = Reshape(target_shape=(1, 1, 10))(dense2)
        return reshape

    main_path_output = main_path(input_tensor=input_layer)
    branch_path_output = branch_path(input_tensor=main_path_output)

    added = keras.layers.Add()([main_path_output, branch_path_output])
    dense3 = Dense(units=512, activation='relu')(added)
    dense4 = Dense(units=256, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model