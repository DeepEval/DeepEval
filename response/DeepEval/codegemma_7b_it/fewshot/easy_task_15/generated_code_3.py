import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv2_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        output_tensor = keras.layers.add([conv2_1, conv2_2])
        return output_tensor

    block_output = block(input_tensor=input_layer)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block_output)
    dropout = Dropout(rate=0.25)(avg_pool)

    for _ in range(2):
        block_output = block(input_tensor=dropout)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block_output)
        dropout = Dropout(rate=0.25)(avg_pool)

    gap = GlobalAveragePooling2D()(dropout)
    flatten = Flatten()(gap)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model