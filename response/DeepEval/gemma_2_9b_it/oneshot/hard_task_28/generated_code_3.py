import keras
from keras.layers import Input, Conv2D, LayerNormalization, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    depthwise_conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', depth_multiplier=1, activation='relu')(input_layer)
    norm_layer = LayerNormalization()(depthwise_conv)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(norm_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Branch path
    branch_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate outputs
    concat_layer = Concatenate()([conv2, branch_conv])

    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model