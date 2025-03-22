import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape
from keras.layers import add

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        return Flatten()(output_tensor)
    
    block1_output = block1(input_layer)

    def block2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return Flatten()(output_tensor)
    
    block2_input = Reshape((1, 1, 64))(block1_output)  # reshape to 4D
    block2_output = block2(block2_input)

    dense1 = Dense(units=128, activation='relu')(block2_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model