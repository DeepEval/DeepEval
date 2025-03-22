import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Define Block 1
    def block1(input_tensor):
        path1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        output_tensor = Concatenate()([Flatten()(path1), Flatten()(path2), Flatten()(path3)])

        return output_tensor

    block1_output = block1(input_layer)

    # Apply a fully connected layer and reshape operation
    dense1 = Dense(units=128, activation='relu')(block1_output)
    reshape_layer = Reshape((128,))(dense1)

    # Define Block 2
    def block2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor

    block2_output = block2(Reshape((1, 1, 128))(reshape_layer))
    bath_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model