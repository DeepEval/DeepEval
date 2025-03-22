import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Conv2D, Reshape, Concatenate, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    pool1_1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten1 = Flatten()(pool1_1)
    dropout1 = Dropout(0.2)(flatten1)

    pool1_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flatten2 = Flatten()(pool1_2)
    dropout2 = Dropout(0.2)(flatten2)

    pool1_3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten3 = Flatten()(pool1_3)
    dropout3 = Dropout(0.2)(flatten3)

    output_block1 = Concatenate()([dropout1, dropout2, dropout3])

    dense1 = Dense(units=64, activation='relu')(output_block1)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense1)

    # Block 2
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    conv2_2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(reshaped)
    conv2_2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2_2)
    
    conv2_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    conv2_3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2_3)
    conv2_3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv2_3)

    conv2_4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(reshaped)
    conv2_4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2_4)

    output_block2 = Concatenate()([conv2_1, conv2_2, conv2_3, conv2_4])

    flatten = Flatten()(output_block2)
    dense2 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model