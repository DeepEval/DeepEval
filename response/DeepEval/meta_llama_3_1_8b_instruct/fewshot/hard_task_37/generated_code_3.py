import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def same_block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        main_path = conv3
        return main_path

    parallel_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    block1_output = same_block(input_layer)
    block2_output = same_block(input_layer)
    parallel_branch_output = parallel_branch

    adding_layer1 = Add()([block1_output, parallel_branch_output])
    adding_layer2 = Add()([block2_output, parallel_branch_output])

    concatenate_output = Concatenate()([adding_layer1, adding_layer2])

    flatten_layer = Flatten()(concatenate_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model