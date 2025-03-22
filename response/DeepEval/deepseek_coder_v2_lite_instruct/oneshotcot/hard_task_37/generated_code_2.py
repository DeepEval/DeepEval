import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        
        path1 = conv1
        path2 = conv2
        path3 = conv3
        
        output_tensor = Add()([path1, path2, path3])
        return output_tensor

    block1_output = block(input_tensor=input_layer)
    block2_output = block(input_tensor=input_layer)

    concatenated_output = Concatenate()([block1_output, block2_output])
    flatten_layer = Flatten()(concatenated_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model