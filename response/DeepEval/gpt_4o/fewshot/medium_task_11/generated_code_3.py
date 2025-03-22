import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Add, Activation, AveragePooling2D, MaxPooling2D, Concatenate, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Channel attention paths
    global_avg_pool = GlobalAveragePooling2D()(conv)
    dense_avg_1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense_avg_2 = Dense(units=64, activation='relu')(dense_avg_1)

    global_max_pool = GlobalMaxPooling2D()(conv)
    dense_max_1 = Dense(units=32, activation='relu')(global_max_pool)
    dense_max_2 = Dense(units=64, activation='relu')(dense_max_1)

    # Combine channel attention paths
    channel_attention = Add()([dense_avg_2, dense_max_2])
    channel_attention = Activation('sigmoid')(channel_attention)

    # Apply channel attention to original features
    channel_scaled_features = Multiply()([conv, channel_attention])

    # Spatial feature extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_scaled_features)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_scaled_features)

    # Concatenate spatial features
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])

    # Combine spatial features with channel-scaled features
    fused_features = Multiply()([channel_scaled_features, spatial_features])

    # Flatten and final dense layer for classification
    flatten_layer = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model