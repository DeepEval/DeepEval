import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Flatten, Multiply
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Two parallel paths for channel-wise feature extraction
    path_gap = GlobalAveragePooling2D()(conv1)
    path_gap = Dense(32, activation='relu')(path_gap)
    path_gap = Dense(32, activation='relu')(path_gap)
    
    path_gmp = GlobalMaxPooling2D()(conv1)
    path_gmp = Dense(32, activation='relu')(path_gmp)
    path_gmp = Dense(32, activation='relu')(path_gmp)
    
    # Combining the outputs of the two paths
    combined = Add()([path_gap, path_gmp])
    activation = keras.activations.relu(combined)
    
    # Channel attention weights
    channel_attention = Dense(conv1.shape[-1], activation='sigmoid')(activation)
    channel_enhanced = Multiply()([conv1, channel_attention])
    
    # Separate average and max pooling for spatial feature extraction
    avg_pool = GlobalAveragePooling2D()(channel_enhanced)
    max_pool = GlobalMaxPooling2D()(channel_enhanced)
    
    # Concatenate spatial features along the channel dimension
    spatial_features = keras.layers.concatenate([avg_pool, max_pool], axis=1)
    
    # Flatten the fused feature map
    flatten = Flatten()(spatial_features)
    
    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model