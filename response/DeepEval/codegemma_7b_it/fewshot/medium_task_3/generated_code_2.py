import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense, Add

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
        return pool

    block1_output = block(input_tensor=input_layer)
    block2_output = block(input_tensor=block1_output)

    # Combine the outputs from both paths with the input
    combined_output = Add()([block1_output, block2_output, input_layer])

    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model