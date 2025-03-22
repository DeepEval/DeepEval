import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # add the input layer with the max pooling layer
    add_layer = Add()([input_layer, max_pooling])

    # batch normalization
    bath_norm = BatchNormalization()(add_layer)

    # flatten the layer
    flatten_layer = Flatten()(bath_norm)

    # dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # dense layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model