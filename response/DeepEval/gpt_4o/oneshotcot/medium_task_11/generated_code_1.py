import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense
from keras.layers import Multiply, Add, Activation, AveragePooling2D, MaxPooling2D, Flatten, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Convolutional Layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Channel Attention Paths
    # Path 1: Global Average Pooling
    gap = GlobalAveragePooling2D()(conv)
    fc1_gap = Dense(units=16, activation='relu')(gap)
    fc2_gap = Dense(units=32, activation='sigmoid')(fc1_gap)
    
    # Path 2: Global Max Pooling
    gmp = GlobalMaxPooling2D()(conv)
    fc1_gmp = Dense(units=16, activation='relu')(gmp)
    fc2_gmp = Dense(units=32, activation='sigmoid')(fc1_gmp)
    
    # Combining Channel Attention Weights
    channel_attention = Add()([fc2_gap, fc2_gmp])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_attended_features = Multiply()([conv, channel_attention])
    
    # Spatial Feature Extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_attended_features)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_attended_features)
    
    # Concatenating Spatial Features
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])
    
    # Combining Spatial and Channel Features
    combined_features = Multiply()([spatial_features, channel_attended_features])
    
    # Flatten and Fully Connected Layer
    flatten = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model