import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    main_path_input = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_input[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_input[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(main_path_input[2])
    main_path_output = Concatenate()([path1, path2, path3])

    # Branch Path
    branch_path_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate and Add
    concat_output = Concatenate()([main_path_output, branch_path_output])
    add_output = Add()([concat_output, branch_path_output])

    # Fully Connected Layers
    flatten_output = Flatten()(add_output)
    dense1 = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model