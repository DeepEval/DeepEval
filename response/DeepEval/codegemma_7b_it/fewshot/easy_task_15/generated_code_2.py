import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
        dropout = Dropout(rate=0.25)(avg_pool)
        return dropout

    block_output = block(input_tensor=input_layer)
    block_output = block(input_tensor=block_output)
    global_avg_pool = GlobalAveragePooling2D()(block_output)
    flatten = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model