import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        x = BatchNormalization()(input_tensor)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        output_tensor = Concatenate()([input_tensor, x])
        return output_tensor

    pathway1 = block(input_layer)
    pathway1 = block(pathway1)
    pathway1 = block(pathway1)

    pathway2 = block(input_layer)
    pathway2 = block(pathway2)
    pathway2 = block(pathway2)

    merged_pathway = Concatenate()([pathway1, pathway2])
    bath_norm = BatchNormalization()(merged_pathway)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model