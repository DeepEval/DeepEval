import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Concatenate, BatchNormalization, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Two parallel paths for channel-wise feature extraction
    # Path 1: Global Average Pooling followed by two fully connected layers
    gap_path1 = GlobalAveragePooling2D()(conv1)
    dense1_path1 = Dense(units=128, activation='relu')(gap_path1)
    dense2_path1 = Dense(units=64, activation='relu')(dense1_path1)
    
    # Path 2: Global Max Pooling followed by two fully connected layers
    gmp_path2 = GlobalMaxPooling2D()(conv1)
    dense1_path2 = Dense(units=128, activation='relu')(gmp_path2)
    dense2_path2 = Dense(units=64, activation='relu')(dense1_path2)
    
    # Combine the outputs from the two paths
    combined = Add()([dense2_path1, dense2_path2])
    activation = keras.activations.relu(combined)
    
    # Channel attention weights
    channel_attention = Dense(units=conv1.shape[-1], activation='sigmoid')(activation)
    channel_enhanced = Multiply()([conv1, channel_attention])
    
    # Separate average and max pooling to extract spatial features
    avg_pool = GlobalAveragePooling2D()(channel_enhanced)
    max_pool = GlobalMaxPooling2D()(channel_enhanced)
    
    # Concatenate the results along the channel dimension
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])
    
    # Flatten the spatial features
    flattened = Flatten()(spatial_features)
    
    # Fully connected layer
    dense3 = Dense(units=256, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model