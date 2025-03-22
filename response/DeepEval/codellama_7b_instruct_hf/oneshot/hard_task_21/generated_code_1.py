import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    main_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Concatenate()([main_path])
    main_path = BatchNormalization()(main_path)

    # Branch path
    branch_path = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    branch_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = BatchNormalization()(branch_path)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)

    # Concat main and branch paths
    output_layer = Concatenate()([main_path, branch_path])

    # Flatten and add fully connected layers
    output_layer = Flatten()(output_layer)
    output_layer = Dense(units=128, activation='relu')(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model