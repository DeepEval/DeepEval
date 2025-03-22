import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_3x3)
        conv_1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_1x1_1)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_1x1_2)
        dropout = Dropout(0.5)(avg_pool)
        return dropout

    block1_output = specialized_block(input_tensor=input_layer)
    block2_output = specialized_block(input_tensor=block1_output)

    global_avg_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(block2_output)
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model