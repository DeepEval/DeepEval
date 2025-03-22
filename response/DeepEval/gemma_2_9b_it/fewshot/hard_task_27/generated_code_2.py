import keras
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    conv_layer = DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    norm_layer = LayerNormalization()(conv_layer)
    
    flatten_1 = Flatten()(norm_layer)
    dense1 = Dense(units=32, activation='relu')(flatten_1)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    add_layer = Add()([input_layer, dense2])
    
    flatten_2 = Flatten()(add_layer)
    dense3 = Dense(units=128, activation='relu')(flatten_2)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model