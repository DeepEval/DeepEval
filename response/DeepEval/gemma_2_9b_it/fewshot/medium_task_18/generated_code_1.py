import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv2_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(pool1)

    concatenated = Concatenate()([conv1_1, conv1_2, conv1_3, conv2_1, conv2_2, conv2_3])

    flatten = Flatten()(concatenated)

    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model