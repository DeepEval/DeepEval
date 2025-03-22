import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Lambda, Multiply, add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv_attention = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    attention_weights = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv_attention)
    contextual_features = Multiply()([input_layer, attention_weights])

    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(contextual_features)
    conv_resize = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
    shortcut = add([conv_resize, contextual_features])

    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(shortcut)

    flatten_layer = Flatten()(max_pooling)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model