import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Dropout, Concatenate
from keras.initializers import RandomNormal

def dl_model():

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block: average pooling layers
    avg_pool_1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    avg_pool_3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    # Flatten and concatenate outputs from pooling layers
    flatten_pool = Concatenate()([Flatten()(avg_pool_1), Flatten()(avg_pool_2), Flatten()(avg_pool_3)])

    # Fully connected layer and reshape
    dense_pool = Dense(units=256, activation='relu')(flatten_pool)
    reshape_pool = Reshape((1, 1, 256))(dense_pool)

    # Second block: parallel paths
    path_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(mean=0., stddev=0.01))(reshape_pool)
    path_2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(mean=0., stddev=0.01))(reshape_pool)
    path_2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(mean=0., stddev=0.01))(path_2_1)
    path_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(mean=0., stddev=0.01))(reshape_pool)
    path_3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(mean=0., stddev=0.01))(path_3)
    path_4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(reshape_pool)
    path_4_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(mean=0., stddev=0.01))(path_4)

    # Dropout layers
    dropout_path_1 = Dropout(rate=0.5)(path_1)
    dropout_path_2 = Dropout(rate=0.5)(path_2_2)
    dropout_path_3 = Dropout(rate=0.5)(path_3_1)
    dropout_path_4 = Dropout(rate=0.5)(path_4_1)

    # Concatenate outputs from all paths
    concat_paths = Concatenate(axis=3)([dropout_path_1, dropout_path_2, dropout_path_3, dropout_path_4])

    # Final fully connected layers
    dense_output = Dense(units=256, activation='relu')(concat_paths)
    output_layer = Dense(units=10, activation='softmax')(dense_output)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model