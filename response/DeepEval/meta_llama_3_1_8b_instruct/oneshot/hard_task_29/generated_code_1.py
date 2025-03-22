import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def block1(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        output_tensor1 = Add()([conv1, conv2])

        branch_path = input_tensor

        return output_tensor1, branch_path
    
    output_tensor1, branch_path = block1(input_layer)
    block2_output, _ = block1(branch_path)
    output_tensor = Add()([output_tensor1, block2_output])

    block3_output = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(output_tensor)
    block3_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block3_output)
    block3_output = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(block3_output)

    bath_norm = BatchNormalization()(block3_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model