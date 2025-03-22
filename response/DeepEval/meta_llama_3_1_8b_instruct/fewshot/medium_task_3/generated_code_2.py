import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_1)

    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
        return max_pooling

    block_output1 = block(input_tensor=pool1)
    block_output2 = block(input_tensor=input_layer)

    adding_layer = Add()([block_output1, block_output2])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model