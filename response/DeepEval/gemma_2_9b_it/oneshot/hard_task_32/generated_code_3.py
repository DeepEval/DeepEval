import keras
from keras.layers import Input, Conv2D, Dropout, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def branch(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = Dropout(0.25)(x) 
        return x

    branch1 = branch(input_layer)
    branch2 = branch(input_layer)
    branch3 = branch(input_layer)

    merged = Concatenate()([branch1, branch2, branch3])
    
    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model