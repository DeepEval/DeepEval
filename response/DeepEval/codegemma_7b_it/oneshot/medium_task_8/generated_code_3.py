import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    main_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    main_input_1 = Lambda(lambda x: x[0])(main_input)
    main_input_2 = Lambda(lambda x: x[1])(main_input)
    main_input_3 = Lambda(lambda x: x[2])(main_input)

    main_conv_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_input_2)
    main_conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv_1)
    main_conv_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv_2)
    main_concat = Concatenate()([main_input_3, main_conv_3])
    main_conv_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_concat)

    branch_conv_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    main_out = add([main_conv_4, branch_conv_1])
    flatten_layer = Flatten()(main_out)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model