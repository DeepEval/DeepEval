import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, AveragePooling2D, MaxPooling2D, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Channel Attention Paths
    avg_pool_path = GlobalAveragePooling2D()(conv_layer)
    avg_dense1 = Dense(units=128, activation='relu')(avg_pool_path)
    avg_dense2 = Dense(units=16, activation='sigmoid')(avg_dense1)

    max_pool_path = GlobalMaxPooling2D()(conv_layer)
    max_dense1 = Dense(units=128, activation='relu')(max_pool_path)
    max_dense2 = Dense(units=16, activation='sigmoid')(max_dense1)

    # Channel Attention
    channel_attention = Add()([avg_dense2, max_dense2])
    channel_attention = tf.keras.activations.sigmoid(channel_attention)  
    channel_weighted_features = conv_layer * channel_attention

    # Spatial Feature Extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_weighted_features)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_weighted_features)

    spatial_features = Concatenate()([avg_pool, max_pool], axis=3)

    # Combine Channel and Spatial Features
    combined_features = channel_weighted_features * spatial_features

    flatten = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model