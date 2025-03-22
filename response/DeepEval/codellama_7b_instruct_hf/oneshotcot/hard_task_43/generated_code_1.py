import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    avg1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(conv1)
    avg2 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(conv2)
    avg3 = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')(conv3)
    concat = Concatenate()([avg1, avg2, avg3])
    bath_norm = BatchNormalization()(concat)
    flatten = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten)

    # Block 2
    conv4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    conv6 = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    avg4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(conv4)
    avg5 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(conv5)
    avg6 = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')(conv6)
    avg7 = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')(conv7)
    concat2 = Concatenate()([avg4, avg5, avg6, avg7])
    bath_norm2 = BatchNormalization()(concat2)
    flatten2 = Flatten()(bath_norm2)
    dense2 = Dense(units=128, activation='relu')(flatten2)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model