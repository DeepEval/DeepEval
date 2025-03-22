import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
        pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

        return pool

    main_path_output = main_path(input_tensor=input_layer)
    batch_norm1 = BatchNormalization()(main_path_output)
    flatten = Flatten()(batch_norm1)

    # Branch path
    def branch_path(input_tensor):
        gp = GlobalAveragePooling2D()(input_tensor)
        fc1 = Dense(units=128, activation='relu')(gp)
        fc2 = Dense(units=64, activation='relu')(fc1)

        return fc2

    branch_path_output = branch_path(input_tensor=flatten)

    # Adding the main path and branch path outputs
    concatenated_output = Concatenate()([main_path_output, branch_path_output])
    batch_norm2 = BatchNormalization()(concatenated_output)
    dense1 = Dense(units=128, activation='relu')(batch_norm2)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model