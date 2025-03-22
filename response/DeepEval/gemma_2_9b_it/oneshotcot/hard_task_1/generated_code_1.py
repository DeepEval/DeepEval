import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, Add, BatchNormalization, Multiply

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1: Channel Attention
    path1 = GlobalAveragePooling2D()(x)
    path1 = Dense(units=128, activation='relu')(path1)
    path1 = Dense(units=3, activation='sigmoid')(path1)  # Output shape matches input channels

    path2 = GlobalMaxPooling2D()(x)
    path2 = Dense(units=128, activation='relu')(path2)
    path2 = Dense(units=3, activation='sigmoid')(path2)  

    channel_attention = Add()([path1, path2])
    channel_attention = Multiply()([x, channel_attention])

    # Block 2: Spatial Attention
    avg_pool = GlobalAveragePooling2D()(channel_attention)
    max_pool = GlobalMaxPooling2D()(channel_attention)
    spatial_features = Concatenate()([avg_pool, max_pool])
    spatial_features = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(spatial_features)
    spatial_features = Multiply()([channel_attention, spatial_features]) 

    # Additional Branch
    additional_branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(spatial_features)

    # Combine outputs
    output = Add()([spatial_features, additional_branch])
    output = Activation('relu')(output)

    # Final Classification
    output_layer = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model