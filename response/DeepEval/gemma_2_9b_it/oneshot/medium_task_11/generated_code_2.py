import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, concatenate, Dense, Add, Activation, AveragePooling2D, MaxPooling2D

def dl_model():  
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Parallel paths with global pooling
    avg_pool = GlobalAveragePooling2D()(x)
    avg_dense1 = Dense(units=128, activation='relu')(avg_pool)
    avg_dense2 = Dense(units=64, activation='relu')(avg_dense1)
    max_pool = GlobalMaxPooling2D()(x)
    max_dense1 = Dense(units=128, activation='relu')(max_pool)
    max_dense2 = Dense(units=64, activation='relu')(max_dense1)

    # Add outputs and apply activation for channel attention
    channel_attention = Add()([avg_dense2, max_dense2])
    channel_attention = Activation('sigmoid')(channel_attention)

    # Apply channel attention to original features
    attended_features = x * channel_attention

    # Spatial pooling
    avg_pool_spatial = AveragePooling2D(pool_size=(2, 2), strides=2)(attended_features)
    max_pool_spatial = MaxPooling2D(pool_size=(2, 2), strides=2)(attended_features)

    # Concatenate spatial features
    spatial_features = concatenate([avg_pool_spatial, max_pool_spatial], axis=3)

    # Combine channel and spatial features
    final_features = attended_features * spatial_features

    # Flatten and output layer
    flatten_layer = Flatten()(final_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model