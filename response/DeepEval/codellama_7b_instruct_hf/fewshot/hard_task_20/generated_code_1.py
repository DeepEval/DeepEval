import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(5, 5), activation='relu')(conv2)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
        output_tensor = Add()([conv1, conv2, conv3])
        return output_tensor

    def branch_path(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        output_tensor = Add()([conv1])
        return output_tensor

    main_path_output = main_path(input_tensor=input_layer)
    branch_path_output = branch_path(input_tensor=input_layer)
    adding_layer = Add()([main_path_output, branch_path_output])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model