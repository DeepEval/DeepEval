import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, concatenate

def dl_model():
    
    input_layer = Input(shape=(32, 32, 64))
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv_1x3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='valid', activation='relu')(conv_1x1)
    conv_3x1 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='valid', activation='relu')(conv_1x1)
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv_1x1)
    concat = concatenate([conv_1x3, conv_3x1, conv_3x3])

    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Add()([concat, branch_path])

    flatten_layer = Flatten()(main_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model