import keras
from keras.layers import Input, Lambda, Concatenate, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = input_layer
    main_path = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(main_path)
    main_path = Concatenate()([
        Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path[0]),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path[1]),
        Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(main_path[2])
    ])
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)

    # Branch path
    branch_path = input_layer
    branch_path = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(branch_path)
    branch_path = Concatenate()([
        Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path[0]),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path[1]),
        Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch_path[2])
    ])
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch_path)

    # Fuse the outputs from both paths
    output = Add()([main_path, branch_path])

    # Flatten and pass through two fully connected layers
    output = Flatten()(output)
    output = Dense(units=64, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model