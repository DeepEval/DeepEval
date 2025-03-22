import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        depthwise = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_multiplier=1)(input_tensor)
        depthwise_dropout = Dropout(0.2)(depthwise)
        pointwise = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_dropout)
        pointwise_dropout = Dropout(0.2)(pointwise)
        return pointwise_dropout

    branch1 = specialized_block(input_layer)
    branch2 = specialized_block(input_layer)
    branch3 = specialized_block(input_layer)

    merged = Concatenate()([branch1, branch2, branch3])

    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model