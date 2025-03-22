import keras
from keras.layers import Input, AveragePooling2D, Concatenate, Dense, Dropout, Flatten, Reshape

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # First block
    avg_pool1 = AveragePooling2D(pool_size=1, strides=1)(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=2, strides=2)(input_layer)
    avg_pool4 = AveragePooling2D(pool_size=4, strides=4)(input_layer)

    concat_avg_pool = Concatenate()([avg_pool1, avg_pool2, avg_pool4])
    flatten_avg_pool = Flatten()(concat_avg_pool)

    # Second block
    path1 = Dense(units=64, activation='relu')(flatten_avg_pool)
    path2 = Dense(units=64, activation='relu')(path1)
    path2 = Dense(units=64, activation='relu')(path2)
    path3 = Dense(units=64, activation='relu')(path1)
    path4 = Dense(units=64, activation='relu')(path1)
    path4 = AveragePooling2D(pool_size=1, strides=1)(path4)
    path4 = Dense(units=64, activation='relu')(path4)

    concat_paths = Concatenate()([path1, path2, path3, path4])

    # Final layers
    flatten_concat = Flatten()(concat_paths)
    reshape_concat = Reshape((1, 1, 256))(flatten_concat)
    dense1 = Dense(units=128, activation='relu')(reshape_concat)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model