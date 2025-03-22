import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Dropout

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    path1 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(split_layer[0])
    path1 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))(path1)
    path1 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(path1)

    path2 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(split_layer[1])
    path2 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))(path2)
    path2 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(path2)

    path3 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(split_layer[2])
    path3 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))(path3)
    path3 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(path3)

    concat_layer = Concatenate()([path1, path2, path3])

    main_path = Lambda(lambda x: x + input_layer)(concat_layer)

    flatten_layer = Flatten()(main_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model