import keras
from keras.layers import Input, Conv2D, Add, AveragePooling2D, GlobalAveragePooling2D, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    add_layer = Add()([conv_3x3, conv_5x5])
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(add_layer)

    global_avg_pool = GlobalAveragePooling2D()(avg_pool)
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    branch1_output = Dense(units=10, activation='softmax')(dense1)
    output_layer = Add()([branch1_output, output_layer])

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model