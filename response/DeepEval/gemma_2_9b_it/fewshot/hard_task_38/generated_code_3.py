import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        x = BatchNormalization()(input_tensor)
        x = ReLU()(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
        x = Concatenate()([input_tensor, x])
        return x

    pathway1 = input_layer
    for _ in range(3):
        pathway1 = block(pathway1)

    pathway2 = input_layer
    for _ in range(3):
        pathway2 = block(pathway2)

    merged_features = Concatenate()([pathway1, pathway2])
    
    flatten_layer = Flatten()(merged_features)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model