import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dense, Activation, Multiply, Dropout, BatchNormalization, Flatten

def dl_model():
  
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Channel-wise attention
    avg_pool_channel = GlobalAveragePooling2D()(max_pooling1)
    dense_avg_pool = Dense(units=128, activation='relu')(avg_pool_channel)
    dense_avg_pool = Dense(units=64, activation='relu')(dense_avg_pool)
    dense_avg_pool = Dense(units=64, activation='sigmoid')(dense_avg_pool)
    reshape_avg_pool = keras.layers.Reshape((1, 1, 64))(dense_avg_pool)
    multiply_avg_pool = Multiply()([max_pooling1, reshape_avg_pool])

    max_pool_channel = GlobalMaxPooling2D()(max_pooling1)
    dense_max_pool = Dense(units=128, activation='relu')(max_pool_channel)
    dense_max_pool = Dense(units=64, activation='relu')(dense_max_pool)
    dense_max_pool = Dense(units=64, activation='sigmoid')(dense_max_pool)
    reshape_max_pool = keras.layers.Reshape((1, 1, 64))(dense_max_pool)
    multiply_max_pool = Multiply()([max_pooling1, reshape_max_pool])

    concat_channel_attention = Concatenate()([multiply_avg_pool, multiply_max_pool])
    batch_norm_attention = BatchNormalization()(concat_channel_attention)
    activation_attention = Activation('relu')(batch_norm_attention)

    # Spatial features
    avg_pool_spatial = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(activation_attention)
    max_pool_spatial = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(activation_attention)

    concat_spatial = Concatenate()([avg_pool_spatial, max_pool_spatial])
    batch_norm_spatial = BatchNormalization()(concat_spatial)
    activation_spatial = Activation('relu')(batch_norm_spatial)

    # Combine features and generate output
    concat_all = Concatenate()([activation_spatial, activation_attention])
    flatten = Flatten()(concat_all)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model