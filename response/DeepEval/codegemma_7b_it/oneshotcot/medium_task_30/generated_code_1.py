import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    pool_1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    pool_2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    pool_3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)
    
    flatten_1 = Flatten()(pool_1)
    flatten_2 = Flatten()(pool_2)
    flatten_3 = Flatten()(pool_3)
    
    concat = concatenate([flatten_1, flatten_2, flatten_3])
    
    dense = Dense(units=10, activation='softmax')(concat)
    
    model = keras.Model(inputs=input_layer, outputs=dense)
    
    return model