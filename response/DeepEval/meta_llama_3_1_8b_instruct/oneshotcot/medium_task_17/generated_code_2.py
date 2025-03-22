import keras
from keras.layers import Input, Reshape, Lambda, Permute, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    reshape = Reshape(target_shape=(32, 32, 3, 1))(input_layer)
    permute = Lambda(lambda x: keras.backend.permute_dimensions(x, (0, 1, 3, 2)))(reshape)
    reshape_back = Reshape(target_shape=(32, 32, 3))(permute)

    dense_output = Dense(units=10, activation='softmax')(reshape_back)

    model = keras.Model(inputs=input_layer, outputs=dense_output)

    return model