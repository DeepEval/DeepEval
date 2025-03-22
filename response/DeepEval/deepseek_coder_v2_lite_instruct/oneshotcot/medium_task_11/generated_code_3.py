import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Multiply, Concatenate, BatchNormalization, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Two parallel paths for channel attention
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    avg_pool = Dense(units=32, activation='relu')(avg_pool)
    avg_pool = Dense(units=32, activation='relu')(avg_pool)
    max_pool = Dense(units=32, activation='relu')(max_pool)
    max_pool = Dense(units=32, activation='relu')(max_pool)
    avg_pool = Dense(units=1, activation='sigmoid')(avg_pool)
    max_pool = Dense(units=1, activation='sigmoid')(max_pool)

    # Combine the outputs of the two paths
    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Multiply()([x, channel_attention])

    # Separate paths for spatial feature extraction
    avg_pool_spatial = GlobalAveragePooling2D()(channel_attention)
    max_pool_spatial = GlobalMaxPooling2D()(channel_attention)
    avg_pool_spatial = Dense(units=32, activation='relu')(avg_pool_spatial)
    max_pool_spatial = Dense(units=32, activation='relu')(max_pool_spatial)

    # Concatenate the outputs of average and max pooling
    spatial_features = Concatenate()([avg_pool_spatial, max_pool_spatial])

    # Flatten the spatial features
    spatial_features = Flatten()(spatial_features)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(spatial_features)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model