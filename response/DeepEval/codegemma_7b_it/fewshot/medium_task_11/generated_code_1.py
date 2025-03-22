import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Flatten, Multiply, concatenate
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Channel Attention
    gap = GlobalAveragePooling2D()(conv)
    gap = Dense(units=16, activation='relu')(gap)
    gap = Dense(units=32, activation='sigmoid')(gap)
    gap = Reshape((1, 1, 32))(gap)

    gmp = GlobalMaxPooling2D()(conv)
    gmp = Dense(units=16, activation='relu')(gmp)
    gmp = Dense(units=32, activation='sigmoid')(gmp)
    gmp = Reshape((1, 1, 32))(gmp)

    attention = Multiply()([gap, gmp])
    channel_attention = Multiply()([attention, conv])

    # Spatial Attention
    avg_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(channel_attention)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(channel_attention)

    avg_pool = Flatten()(avg_pool)
    max_pool = Flatten()(max_pool)

    concat = concatenate([avg_pool, max_pool])
    spatial_attention = Dense(units=16, activation='relu')(concat)
    spatial_attention = Dense(units=32, activation='sigmoid')(spatial_attention)

    spatial_attention = Reshape((4, 4, 32))(spatial_attention)

    fused_features = Multiply()([spatial_attention, channel_attention])

    # Classification
    flatten = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model