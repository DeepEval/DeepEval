import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, multiply, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    global_avg_pool = GlobalAveragePooling2D()(input_layer)

    def block(input_tensor):

        dense1 = Dense(units=32, activation='relu')(input_tensor)
        dense2 = Dense(units=32, activation='relu')(dense1)
        dense3 = Dense(units=32, activation='relu')(dense2)
        dense4 = Dense(units=32, activation='relu')(dense3)
        weights = Concatenate()([dense1, dense2, dense3, dense4])
        weights = Reshape((32,))(weights)
        output_tensor = multiply([input_tensor, weights])

        return output_tensor
    
    block_output = block(input_tensor=global_avg_pool)
    flatten_layer = Flatten()(block_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model