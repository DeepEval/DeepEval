import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        gap = GlobalAveragePooling2D()(input_tensor)
        fc1 = Dense(units=input_tensor.shape[-1], activation='relu')(gap)
        fc2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(fc1)
        fc2_reshape = Reshape((1, 1, input_tensor.shape[-1]))(fc2)
        weighted_input = Multiply()([input_tensor, fc2_reshape])
        return weighted_input

    branch1_output = block(input_tensor=input_layer)
    branch2_output = block(input_tensor=input_layer)

    concat = Concatenate()([branch1_output, branch2_output])
    flatten = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model