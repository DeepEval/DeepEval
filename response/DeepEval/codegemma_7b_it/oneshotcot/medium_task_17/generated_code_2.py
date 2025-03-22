import keras
from keras.layers import Input, Reshape, Permute, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    reshaped_tensor = Reshape((32, 32, 3, 3))(input_layer)
    swapped_tensor = Permute((0, 1, 3, 2))(reshaped_tensor)
    reshaped_tensor_back = Reshape((32, 32, 3))(swapped_tensor)
    output_layer = Dense(units=10, activation='softmax')(reshaped_tensor_back)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model