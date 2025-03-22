import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # First block: Average pooling layers
    avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    # Flatten and concatenate outputs from pooling layers
    pooled_features = Concatenate()([Flatten()(avg_pool_1x1), Flatten()(avg_pool_2x2), Flatten()(avg_pool_4x4)])

    # Second block: Multi-path feature extraction
    path_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pooled_features)
    path_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pooled_features)
    path_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path_2)
    path_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pooled_features)
    path_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path_3)
    path_4 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(pooled_features)
    path_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path_4)

    # Dropout layers
    path_1 = Dropout(0.5)(path_1)
    path_2 = Dropout(0.5)(path_2)
    path_3 = Dropout(0.5)(path_3)
    path_4 = Dropout(0.5)(path_4)

    # Concatenate outputs from all paths
    concat_features = Concatenate()([path_1, path_2, path_3, path_4])

    # Fully connected layers
    flatten_features = Flatten()(concat_features)
    dense_1 = Dense(units=128, activation='relu')(flatten_features)
    dense_2 = Dense(units=64, activation='relu')(dense_1)
    output_layer = Dense(units=10, activation='softmax')(dense_2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model