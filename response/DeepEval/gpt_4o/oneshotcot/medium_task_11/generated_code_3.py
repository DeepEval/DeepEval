import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Multiply, AveragePooling2D, MaxPooling2D, Concatenate, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Channel dimension feature extraction
    # Path 1: Global Average Pooling
    gap = GlobalAveragePooling2D()(conv)
    fc_gap1 = Dense(units=16, activation='relu')(gap)
    fc_gap2 = Dense(units=32, activation='sigmoid')(fc_gap1)  # Sigmoid to generate attention weights

    # Path 2: Global Max Pooling
    gmp = GlobalMaxPooling2D()(conv)
    fc_gmp1 = Dense(units=16, activation='relu')(gmp)
    fc_gmp2 = Dense(units=32, activation='sigmoid')(fc_gmp1)  # Sigmoid to generate attention weights

    # Combine channel attention weights
    channel_attention = Add()([fc_gap2, fc_gmp2])
    channel_weighted_features = Multiply()([conv, channel_attention])

    # Spatial feature extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_weighted_features)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_weighted_features)

    # Concatenate spatial features
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])

    # Combine spatial and channel features
    combined_features = Multiply()([spatial_features, channel_weighted_features])

    # Flatten and final dense layer for classification
    flatten = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# You can now create the model and compile it
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])