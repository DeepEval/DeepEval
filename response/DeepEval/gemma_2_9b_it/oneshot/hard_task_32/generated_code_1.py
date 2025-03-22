import keras
from keras.layers import Input, Conv2D, Dropout, Concatenate, Flatten, Dense

def dl_model():  

    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        depthwise = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', 
                           depthwise=True)(input_tensor)
        dropout1 = Dropout(rate=0.2)(depthwise)
        pointwise = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
        dropout2 = Dropout(rate=0.2)(pointwise)
        return dropout2

    branch1 = specialized_block(input_layer)
    branch2 = specialized_block(input_layer)
    branch3 = specialized_block(input_layer)

    concatenated = Concatenate()([branch1, branch2, branch3])
    flatten = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model