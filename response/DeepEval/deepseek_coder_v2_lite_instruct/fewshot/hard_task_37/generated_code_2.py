import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        path1 = conv3
        path2 = conv3
        path3 = conv3
        return path1, path2, path3

    block1_path1, block1_path2, block1_path3 = block(input_layer)
    block2_path1, block2_path2, block2_path3 = block(input_layer)

    branch_input = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path1, branch_path2, branch_path3 = block(branch_input)

    adding_layer1 = Add()([block1_path1, block2_path1, branch_path1])
    adding_layer2 = Add()([block1_path2, block2_path2, branch_path2])
    adding_layer3 = Add()([block1_path3, block2_path3, branch_path3])

    concatenated_layer = Concatenate()([adding_layer1, adding_layer2, adding_layer3])
    flatten_layer = Flatten()(concatenated_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model