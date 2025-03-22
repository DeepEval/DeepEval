import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(main_path[0])
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(main_path[1])
    main_path = Concatenate()([main_path[0], main_path[1], main_path[2]])
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)

    # Branch path
    branch_path = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path[0])
    branch_path = Concatenate()([branch_path[0], branch_path[1], branch_path[2]])
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)

    # Fuse main and branch paths
    output = Concatenate()([main_path, branch_path])
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model