import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = BatchNormalization()(conv)
        return conv
    
    def pathway(input_tensor):
        for _ in range(3):
            input_tensor = block(input_tensor)
        return input_tensor
    
    pathway1 = pathway(input_layer)
    pathway2 = pathway(input_layer)
    
    merged = Concatenate(axis=-1)([pathway1, pathway2])
    
    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model