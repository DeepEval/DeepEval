import keras
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Multiply, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=512, activation='relu')(avg_pool)
        dense2 = Dense(units=512, activation='relu')(dense1)
        weights = Dense(shape=input_tensor.shape[1::-1], activation='linear')(dense2)
        output_tensor = Multiply()([input_tensor, weights] )
        return output_tensor

    branch1 = block(input_tensor=input_layer)
    branch2 = block(input_tensor=input_layer)

    concat = Concatenate()([branch1, branch2])
    flatten = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model