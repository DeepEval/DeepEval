import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense, Dropout, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block
    avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    avg_pool_flat = Concatenate(axis=1)([Flatten()(avg_pool_1x1), Flatten()(avg_pool_2x2), Flatten()(avg_pool_4x4)])
    avg_pool_dense = Dense(units=100, activation='relu')(avg_pool_flat)
    avg_pool_reshape = Reshape(target_shape=(1, 100))(avg_pool_dense)

    # Second block
    path_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool_reshape)
    path_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool_reshape)
    path_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path_2)
    path_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool_reshape)
    path_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path_3)
    path_4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(avg_pool_reshape)
    path_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path_4)

    path_1 = Dropout(0.2)(path_1)
    path_2 = Dropout(0.2)(path_2)
    path_3 = Dropout(0.2)(path_3)
    path_4 = Dropout(0.2)(path_4)

    concat = Concatenate(axis=3)([path_1, path_2, path_3, path_4])
    concat_flat = Flatten()(concat)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(concat_flat)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model