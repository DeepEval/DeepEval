import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, AveragePooling2D, MaxPooling2D, Concatenate, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Global average pooling path
    gap = GlobalAveragePooling2D()(conv)
    fc1_gap = Dense(units=16, activation='relu')(gap)
    fc2_gap = Dense(units=32, activation='sigmoid')(fc1_gap)
    
    # Global max pooling path
    gmp = GlobalMaxPooling2D()(conv)
    fc1_gmp = Dense(units=16, activation='relu')(gmp)
    fc2_gmp = Dense(units=32, activation='sigmoid')(fc1_gmp)
    
    # Channel attention weights
    channel_attention_weights = Multiply()([fc2_gap, fc2_gmp])
    
    # Apply channel attention to the original features
    channel_attention_applied = Multiply()([conv, channel_attention_weights])
    
    # Spatial feature extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_attention_applied)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_attention_applied)
    
    # Concatenate spatial features
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])
    
    # Fuse channel and spatial features
    fused_features = Multiply()([channel_attention_applied, spatial_features])
    
    # Flatten and output layer
    flatten_layer = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model