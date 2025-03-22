import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dense, Multiply, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Channel attention path
    avg_pool = GlobalAveragePooling2D()(conv)
    avg_pool = Dense(units=16, activation='relu')(avg_pool)
    avg_pool = Dense(units=64, activation='sigmoid')(avg_pool)

    max_pool = GlobalMaxPooling2D()(conv)
    max_pool = Dense(units=16, activation='relu')(max_pool)
    max_pool = Dense(units=64, activation='sigmoid')(max_pool)

    concat = Concatenate()([avg_pool, max_pool])
    channel_attention = Multiply()([conv, concat])

    # Spatial attention path
    avg_pool_spatial = AveragePooling2D()(channel_attention)
    max_pool_spatial = MaxPooling2D()(channel_attention)

    avg_pool_spatial = Flatten()(avg_pool_spatial)
    max_pool_spatial = Flatten()(max_pool_spatial)

    avg_pool_spatial = Dense(units=16, activation='relu')(avg_pool_spatial)
    max_pool_spatial = Dense(units=16, activation='relu')(max_pool_spatial)

    concat_spatial = Concatenate()([avg_pool_spatial, max_pool_spatial])
    concat_spatial = Dense(units=64, activation='sigmoid')(concat_spatial)
    concat_spatial = Reshape((32, 32, 64))(concat_spatial)

    # Combine channel and spatial features
    fused_features = Multiply()([channel_attention, concat_spatial])

    # Output layer
    output_layer = Flatten()(fused_features)
    output_layer = Dense(units=64, activation='relu')(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model