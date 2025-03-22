import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Reshape, Multiply

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    global_pooling = AveragePooling2D(pool_size=(4, 4), strides=1, padding='valid')(max_pooling)
    flatten_layer = Flatten()(global_pooling)
    dense1 = Dense(units=32, activation='relu')(flatten_layer)
    weight_matrix = Dense(units=32, activation='softmax')(dense1)
    reshaped_weights = Reshape((1, 1, 32))(weight_matrix)
    main_output = Multiply()([reshaped_weights, max_pooling])
    main_output = Flatten()(main_output)
    dense2 = Dense(units=128, activation='relu')(main_output)
    branch_output = Flatten()(input_layer)
    output = keras.layers.Add()([branch_output, dense2])
    output_layer = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model