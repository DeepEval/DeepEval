import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    branch1_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1_output)
    branch1_output = Dropout(0.2)(branch1_output)

    branch2_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_output = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2_output)
    branch2_output = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2_output)
    branch2_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_output)
    branch2_output = Dropout(0.2)(branch2_output)

    branch3_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3_output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(branch3_output)
    branch3_output = Dropout(0.2)(branch3_output)

    merged_output = Concatenate()([branch1_output, branch2_output, branch3_output])
    bath_norm = BatchNormalization()(merged_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model