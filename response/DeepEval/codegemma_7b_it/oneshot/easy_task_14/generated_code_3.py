import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    global_avg_pool = GlobalAveragePooling2D()(input_layer)

    dense1 = Dense(units=10, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)
    dense3 = Dense(units=32, activation='relu')(dense2)
    dense4 = Dense(units=32, activation='relu')(dense3)
    dense5 = Dense(units=10, activation='relu')(dense4)

    reshaped_weights = keras.backend.reshape(dense5, (32, 32, 3, 1))
    conv_output = keras.backend.conv2d(input_layer, reshaped_weights, padding='same')

    output_layer = Flatten()(conv_output)
    dense6 = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=dense6)

    return model