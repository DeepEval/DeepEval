import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def same_block(input_tensor):
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(64, activation='relu')(global_avg_pool)
        dense2 = Dense(32, activation='relu')(dense1)
        weights = Dense(3, activation='linear')(dense2)
        weights = Reshape(target_shape=(1, 1, 3))(weights)
        output_tensor = Multiply()([input_tensor, weights])
        return output_tensor

    branch1_output = same_block(input_tensor=input_layer)
    branch2_output = same_block(input_tensor=input_layer)
    output_tensor = Concatenate()([branch1_output, branch2_output])
    flatten_layer = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model