import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense, Dropout, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block1(input_tensor):
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path1 = Flatten()(path1)
        path1 = Dropout(0.5)(path1)

        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path2 = Flatten()(path2)
        path2 = Dropout(0.5)(path2)

        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        path3 = Flatten()(path3)
        path3 = Dropout(0.5)(path3)

        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    block1_output = block1(input_layer)

    # Between block 1 and block 2
    dense_intermediate = Dense(units=512, activation='relu')(block1_output)

    # Reshape for 4D tensor
    reshape_layer = Reshape((8, 8, 8))(dense_intermediate)  # Assuming reshaping to a compatible shape

    # Block 2
    def block2(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)

        path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block2_output = block2(reshape_layer)

    # Final fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model