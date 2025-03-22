import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_1)
    dropout_1 = Dropout(0.2)(max_pooling_1)

    conv_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(dropout_1)
    max_pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_2)
    dropout_2 = Dropout(0.2)(max_pooling_2)

    conv_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(dropout_2)
    max_pooling_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_3)
    dropout_3 = Dropout(0.2)(max_pooling_3)

    flatten_1 = Flatten()(dropout_1)
    flatten_2 = Flatten()(dropout_2)
    flatten_3 = Flatten()(dropout_3)

    concatenate_1 = Concatenate()([flatten_1, flatten_2, flatten_3])

    # Block 2
    conv_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenate_1)
    max_pooling_4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_4)
    dropout_4 = Dropout(0.2)(max_pooling_4)

    conv_5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout_4)
    max_pooling_5 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_5)
    dropout_5 = Dropout(0.2)(max_pooling_5)

    conv_6 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(dropout_5)
    max_pooling_6 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_6)
    dropout_6 = Dropout(0.2)(max_pooling_6)

    flatten_4 = Flatten()(dropout_4)
    flatten_5 = Flatten()(dropout_5)
    flatten_6 = Flatten()(dropout_6)

    concatenate_2 = Concatenate()([flatten_4, flatten_5, flatten_6])

    dense_1 = Dense(units=128, activation='relu')(concatenate_2)
    dense_2 = Dense(units=64, activation='relu')(dense_1)
    output_layer = Dense(units=10, activation='softmax')(dense_2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model