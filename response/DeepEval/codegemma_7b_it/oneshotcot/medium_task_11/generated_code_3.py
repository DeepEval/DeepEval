import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dense, Flatten, Activation, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Channel-wise attention
    avg_pool = GlobalAveragePooling2D()(conv)
    max_pool = GlobalMaxPooling2D()(conv)

    # Fully connected layers for global average pooling path
    avg_dense1 = Dense(units=64, activation='relu')(avg_pool)
    avg_dense2 = Dense(units=32, activation='relu')(avg_dense1)

    # Fully connected layers for global max pooling path
    max_dense1 = Dense(units=64, activation='relu')(max_pool)
    max_dense2 = Dense(units=32, activation='relu')(max_dense1)

    # Concatenate outputs from both paths
    concat = Concatenate()([avg_dense2, max_dense2])

    # Activation function to generate channel attention weights
    attention_weights = Activation('sigmoid')(concat)

    # Element-wise multiplication to apply attention weights
    channel_attention = Multiply()([conv, attention_weights])

    # Spatial features extraction
    avg_pool_spatial = AveragePooling2D()(channel_attention)
    max_pool_spatial = MaxPooling2D()(channel_attention)

    # Concatenate spatial features
    spatial_concat = Concatenate()([avg_pool_spatial, max_pool_spatial])

    # Spatial feature processing
    spatial_dense1 = Dense(units=64, activation='relu')(spatial_concat)
    spatial_dense2 = Dense(units=32, activation='relu')(spatial_dense1)

    # Element-wise multiplication of spatial and channel features
    fused_features = Multiply()([channel_attention, spatial_dense2])

    # Flatten and fully connected layers
    flatten = Flatten()(fused_features)
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model