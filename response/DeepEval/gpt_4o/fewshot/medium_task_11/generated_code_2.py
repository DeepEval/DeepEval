import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, AveragePooling2D, MaxPooling2D, Flatten, Concatenate, Activation

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv_initial = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Channel attention via two parallel paths
    # Path 1: Global Average Pooling
    gap = GlobalAveragePooling2D()(conv_initial)
    dense1_1 = Dense(units=64, activation='relu')(gap)
    dense1_2 = Dense(units=64, activation='sigmoid')(dense1_1)
    
    # Path 2: Global Max Pooling
    gmp = GlobalMaxPooling2D()(conv_initial)
    dense2_1 = Dense(units=64, activation='relu')(gmp)
    dense2_2 = Dense(units=64, activation='sigmoid')(dense2_1)
    
    # Adding the outputs from both paths
    add_paths = keras.layers.Add()([dense1_2, dense2_2])
    
    # Apply channel attention weights
    channel_attention = Multiply()([conv_initial, add_paths])
    
    # Spatial feature extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_attention)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_attention)
    
    # Concatenate spatial features along the channel dimension
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])
    
    # Combine channel and spatial features
    combined_features = Multiply()([channel_attention, spatial_features])
    
    # Flatten the features and final dense layer for classification
    flatten = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model