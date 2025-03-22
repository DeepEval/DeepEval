import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    concat = Concatenate()([conv1, conv2])
    global_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(concat)
    flatten = Flatten()(global_pool)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    softmax = Dense(units=10, activation='softmax')(dense2)
    attention = Concatenate()([conv1, conv2])
    attention_weights = Dense(units=10, activation='softmax')(attention)
    output = attention_weights * softmax
    model = keras.Model(inputs=input_layer, outputs=output)
    return model