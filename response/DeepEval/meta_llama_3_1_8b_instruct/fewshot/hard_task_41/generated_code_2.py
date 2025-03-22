import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dropout, Concatenate, Dense, Reshape
from keras.layers import Conv2D, Add

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        flatten1 = Dropout(0.2)(flatten1) # Regularize using dropout operation

        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        flatten2 = Dropout(0.2)(flatten2) # Regularize using dropout operation

        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        flatten3 = Dropout(0.2)(flatten3) # Regularize using dropout operation

        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    def block_2(input_tensor):
        inputs_groups1 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        inputs_groups2 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        inputs_groups2 = Conv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups2)

        inputs_groups3 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        inputs_groups3 = Conv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups3)
        inputs_groups3 = Conv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups3)

        inputs_groups4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        inputs_groups4 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups4)

        output_tensor = Concatenate()([inputs_groups1, inputs_groups2, inputs_groups3, inputs_groups4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 64))(dense)
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model