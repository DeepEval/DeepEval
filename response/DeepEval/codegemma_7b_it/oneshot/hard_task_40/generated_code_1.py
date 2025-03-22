import keras
from keras.layers import Input, AveragePooling2D, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # First block: Average pooling layers
    avg_pool_1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    avg_pool_3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    # Flatten and concatenate outputs of average pooling layers
    avg_pool_output = Concatenate()([Flatten()(avg_pool_1), Flatten()(avg_pool_2), Flatten()(avg_pool_3)])

    # Fully connected layer and reshape
    dense_1 = Dense(units=128, activation='relu')(avg_pool_output)
    reshape = Reshape((4, 4, 32))(dense_1)

    # Second block: Parallel paths
    path_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape)
    path_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape)
    path_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path_2)
    path_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path_2)
    path_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape)
    path_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path_3)
    path_4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(reshape)
    path_4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path_4)

    # Dropout and concatenate outputs of parallel paths
    path_1 = Dropout(rate=0.3)(path_1)
    path_2 = Dropout(rate=0.3)(path_2)
    path_3 = Dropout(rate=0.3)(path_3)
    path_4 = Dropout(rate=0.3)(path_4)
    concat = Concatenate()([path_1, path_2, path_3, path_4])

    # Final fully connected layers
    dense_2 = Dense(units=10, activation='softmax')(concat)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=dense_2)

    return model