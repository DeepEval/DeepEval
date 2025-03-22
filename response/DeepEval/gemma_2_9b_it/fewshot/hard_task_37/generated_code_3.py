import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv2)

        main_path1 = conv1
        main_path2 = conv2
        main_path3 = conv3
        
        branch_path = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(input_tensor)

        output_tensor = Add()([main_path1, main_path2, main_path3, branch_path])
        return output_tensor

    block1_output = block(input_layer)
    block2_output = block(block1_output)

    concatenated_output = Concatenate()([block1_output, block2_output])
    flatten_layer = Flatten()(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model