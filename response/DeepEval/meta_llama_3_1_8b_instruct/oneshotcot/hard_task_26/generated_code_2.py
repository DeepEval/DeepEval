import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv_initial = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def branch1(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return path1

    def branch2(input_tensor):
        path2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(path2)
        return path2

    def branch3(input_tensor):
        path3 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(path3)
        return path3

    branch1_output = branch1(conv_initial)
    branch2_output = branch2(conv_initial)
    branch3_output = branch3(conv_initial)

    output_tensor = Concatenate()([branch1_output, branch2_output, branch3_output])
    output_tensor = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)

    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_initial)

    added_output = Add()([output_tensor, branch_path])

    batch_norm = BatchNormalization()(added_output)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model