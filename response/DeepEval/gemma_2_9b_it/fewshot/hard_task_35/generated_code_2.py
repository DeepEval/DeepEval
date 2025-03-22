import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Concatenate, Flatten

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        pooled_output = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=256, activation='relu')(pooled_output)
        dense2 = Dense(units=3, activation='relu')(dense1)
        reshape_layer = Reshape(target_shape=(32, 32, 3))(dense2)
        output_tensor = input_tensor * reshape_layer
        return output_tensor

    branch1_output = block(input_layer)
    branch2_output = block(input_layer)
    concatenated_output = Concatenate()([branch1_output, branch2_output])
    flatten_layer = Flatten()(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model