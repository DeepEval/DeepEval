import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, AveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
    avg_pool = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(input_layer)
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)

    branch1_output = Dropout(0.2)(conv1)
    branch2_output = Dropout(0.2)(Concatenate()([conv2, conv1]))
    branch3_output = Dropout(0.2)(Concatenate()([conv3_2, conv3_1, conv3]))
    branch4_output = Dropout(0.2)(conv4)

    output_tensor = Concatenate()([branch1_output, branch2_output, branch3_output, branch4_output])

    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense1 = Dropout(0.2)(dense1)
    dense2 = Dense(units=64, activation='relu')(dense1)
    dense2 = Dropout(0.2)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model