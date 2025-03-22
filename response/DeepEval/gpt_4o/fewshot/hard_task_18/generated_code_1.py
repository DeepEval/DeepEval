import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Multiply, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        output_tensor = Add()([input_tensor, avg_pool])
        return output_tensor

    # Second block
    def block_2(input_tensor):
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=32, activation='relu')(global_avg_pool)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
        reshaped = keras.layers.Reshape((1, 1, input_tensor.shape[-1]))(dense2)
        scaled = Multiply()([input_tensor, reshaped])
        return scaled

    # Construct the model
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model