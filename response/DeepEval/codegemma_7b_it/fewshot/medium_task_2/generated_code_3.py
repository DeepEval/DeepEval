import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, concatenate

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    def main_path(input_tensor):
        conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
        main_path_output = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)
        return main_path_output

    def branch_path(input_tensor):
        conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch_path_output = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_1)
        return branch_path_output

    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)
    combined_features = concatenate([main_path_output, branch_path_output])

    flatten_layer = Flatten()(combined_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model