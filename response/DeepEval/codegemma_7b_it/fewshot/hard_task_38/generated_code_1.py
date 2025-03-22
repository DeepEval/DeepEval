import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        batch_norm = BatchNormalization()(input_tensor)
        activation = keras.layers.ReLU()(batch_norm)
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(activation)
        concat = Concatenate()([input_tensor, conv])
        return concat

    block_output_1 = block(input_tensor=input_layer)
    block_output_2 = block(input_tensor=block_output_1)
    block_output_3 = block(input_tensor=block_output_2)

    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block_output_3)
    flatten = Flatten()(max_pooling)
    dense1 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model