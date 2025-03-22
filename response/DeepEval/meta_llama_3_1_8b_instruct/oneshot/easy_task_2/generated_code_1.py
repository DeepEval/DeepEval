import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(224, 224, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pooling1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pooling1)
    avg_pooling2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pooling2)
    avg_pooling3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    flatten_layer = Flatten()(avg_pooling3)

    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    output_layer = Dense(units=1000, activation='softmax')(dropout2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model