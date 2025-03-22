import keras
from keras.layers import Input, Conv2D, Multiply, Add, Lambda, Reshape, Dense, GlobalAveragePooling2D, LayerNormalization, ReLU, concatenate
from keras import backend as K
from keras.layers import AveragePooling2D
import numpy as np

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Step 1: Generate attention weights
    conv_att = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    softmax_att = Dense(32, activation='softmax')(conv_att)
    reshape_att = Reshape((32, 32, 1))(softmax_att)
    attention_weights = Multiply()([reshape_att, input_layer])

    # Step 2: Reduce dimensionality
    conv_reduce = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(attention_weights)
    layer_norm = LayerNormalization()(conv_reduce)
    activation = ReLU()(layer_norm)
    conv_restore = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(activation)

    # Step 3: Add original input and processed output
    add_layer = Add()([input_layer, conv_restore])

    # Step 4: Flatten and classification
    flatten_layer = Flatten()(add_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model