import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras import backend as K

def dl_model():    

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Lambda(lambda x: K.split(x, 3, axis=-1))(input_layer)
    main_path_output = []
    for i in range(3):
        main_path_output.append(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path[i]))

    main_path_output = Concatenate(axis=-1)(main_path_output)
    main_path_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_output)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion
    combined_output = keras.layers.Add()([main_path_output, branch_path])

    # Output layer
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model