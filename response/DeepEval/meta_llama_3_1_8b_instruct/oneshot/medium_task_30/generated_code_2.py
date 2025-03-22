import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    avg_pooling1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    avg_pooling2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    avg_pooling3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)

    def fusion_block(input_tensor):

        output_tensor = Concatenate()([avg_pooling1, avg_pooling2, avg_pooling3])

        return output_tensor
        
    fusion_output = fusion_block(input_tensor=input_layer)
    flatten_layer = Flatten()(fusion_output)
    fusion = Concatenate()([flatten_layer])
    fusion_output = Flatten()(fusion)

    dense1 = Dense(units=128, activation='relu')(fusion_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model