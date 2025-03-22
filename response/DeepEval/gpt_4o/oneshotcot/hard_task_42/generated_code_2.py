import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, Reshape, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block1(input_tensor):
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)

        path1_flatten = Flatten()(path1)
        path2_flatten = Flatten()(path2)
        path3_flatten = Flatten()(path3)

        path1_dropout = Dropout(0.5)(path1_flatten)
        path2_dropout = Dropout(0.5)(path2_flatten)
        path3_dropout = Dropout(0.5)(path3_flatten)

        output_tensor = Concatenate()([path1_dropout, path2_dropout, path3_dropout])
        
        return output_tensor

    block1_output = block1(input_layer)

    # Fully connected layer and Reshape
    fc = Dense(units=128, activation='relu')(block1_output)
    reshape = Reshape(target_shape=(4, 4, 8))(fc)

    # Block 2
    def block2(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)

        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu', padding='same')(path2)
        path2 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(path2)

        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu', padding='same')(path3)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu', padding='same')(path3)

        path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path4)

        output_tensor = Concatenate(axis=-1)([path1, path2, path3, path4])
        
        return output_tensor

    block2_output = block2(reshape)

    # Final fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model