import keras
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Concatenate, Add, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Extracting spatial features with a 7x7 depthwise separable convolutional layer and layer normalization
    conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same')(input_layer)
    conv = LayerNormalization()(conv)
    conv = keras.layers.Activation('relu')(conv)

    # Channel-wise feature transformation using two fully connected layers
    conv = Dense(units=128, activation='relu')(conv)
    conv = Dense(units=128, activation='relu')(conv)
    conv = keras.layers.Activation('relu')(conv)

    # Combine the original input with the processed features through an addition operation
    add_input = Add()([input_layer, conv])
    concat_layer = Concatenate()([input_layer, conv])

    # Final two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(concat_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model