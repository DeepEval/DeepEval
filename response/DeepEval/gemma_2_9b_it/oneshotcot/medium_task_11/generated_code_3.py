import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, AveragePooling2D, MaxPooling2D, Activation

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Channel Attention Paths
    avg_pool = GlobalAveragePooling2D()(conv)
    avg_dense1 = Dense(units=128, activation='relu')(avg_pool)
    avg_dense2 = Dense(units=64, activation='relu')(avg_dense1)
    avg_output = Dense(32, activation='sigmoid')(avg_dense2)

    max_pool = GlobalMaxPooling2D()(conv)
    max_dense1 = Dense(units=128, activation='relu')(max_pool)
    max_dense2 = Dense(units=64, activation='relu')(max_dense1)
    max_output = Dense(32, activation='sigmoid')(max_dense2)

    # Channel Attention Fusion
    channel_attention = Add()([avg_output, max_output])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_attention = tf.multiply(conv, channel_attention) 

    # Spatial Feature Extraction
    avg_pool_spatial = AveragePooling2D(pool_size=(2, 2))(channel_attention)
    max_pool_spatial = MaxPooling2D(pool_size=(2, 2))(channel_attention)
    spatial_features = Concatenate(axis=3)([avg_pool_spatial, max_pool_spatial])

    # Combine Channel and Spatial Features
    combined_features = tf.multiply(channel_attention, spatial_features)

    # Final Processing
    flatten_layer = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model