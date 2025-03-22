import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        output_tensor = keras.layers.add([conv1, conv2])
        return output_tensor

    block1_output = block1(input_tensor=input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(block1_output)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1_output)
    max_pooling3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(block1_output)

    max_pooling_concat = keras.layers.concatenate([max_pooling1, max_pooling2, max_pooling3])
    flatten_layer = Flatten()(max_pooling_concat)

    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model