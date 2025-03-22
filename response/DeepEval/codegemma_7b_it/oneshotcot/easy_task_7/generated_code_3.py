import keras
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv_main1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    drop_main1 = Dropout(rate=0.2)(conv_main1)
    conv_main2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(drop_main1)
    drop_main2 = Dropout(rate=0.2)(conv_main2)
    conv_main3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid')(drop_main2)

    # Branch path
    conv_branch = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine paths
    concat_layer = Add()([conv_main3, conv_branch])
    bath_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(bath_norm)
    dense = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model