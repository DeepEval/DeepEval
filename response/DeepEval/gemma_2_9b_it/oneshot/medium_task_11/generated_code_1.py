import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, Add, AveragePooling2D, MaxPooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    avg_pool = GlobalAveragePooling2D()(conv)
    avg_fc1 = Dense(units=128, activation='relu')(avg_pool)
    avg_fc2 = Dense(units=64, activation='relu')(avg_fc1)

    max_pool = GlobalMaxPooling2D()(conv)
    max_fc1 = Dense(units=128, activation='relu')(max_pool)
    max_fc2 = Dense(units=64, activation='relu')(max_fc1)

    channel_attention = Add()([avg_fc2, max_fc2])
    channel_attention = Activation('sigmoid')(channel_attention)
    
    # Element-wise multiplication for channel attention
    channel_features = conv * channel_attention

    # Spatial feature extraction
    avg_pool_spatial = AveragePooling2D(pool_size=(2, 2))(channel_features)
    max_pool_spatial = MaxPooling2D(pool_size=(2, 2))(channel_features)

    # Concatenate spatial features
    spatial_features = Concatenate()([avg_pool_spatial, max_pool_spatial])

    # Element-wise multiplication for fusing channel and spatial features
    fused_features = channel_features * spatial_features

    flatten = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model