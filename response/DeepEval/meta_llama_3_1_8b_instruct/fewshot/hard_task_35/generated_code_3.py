import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def same_block(input_tensor):
        avgpool = GlobalAveragePooling2D()(input_tensor)
        fc1 = Dense(units=input_tensor.shape[-1], activation='relu')(avgpool)
        fc2 = Dense(units=input_tensor.shape[-1], activation='relu')(fc1)
        weights = Reshape(target_shape=input_tensor.shape[1:])(fc2)
        output_tensor = Multiply()([input_tensor, weights])
        return output_tensor

    branch1_output = same_block(input_layer)
    branch2_output = same_block(input_layer)

    concat_output = Concatenate()([branch1_output, branch2_output])

    flatten_layer = Flatten()(concat_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model