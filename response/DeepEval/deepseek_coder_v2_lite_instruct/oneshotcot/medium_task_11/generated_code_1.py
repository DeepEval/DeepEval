import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Two parallel paths for channel dimension processing
    path_avg = GlobalAveragePooling2D()(conv1)
    path_avg = Dense(32, activation='relu')(path_avg)
    path_avg = Dense(16, activation='relu')(path_avg)
    
    path_max = GlobalMaxPooling2D()(conv1)
    path_max = Dense(32, activation='relu')(path_max)
    path_max = Dense(16, activation='relu')(path_max)
    
    # Merge the outputs of the two paths
    channel_attention = Add()([path_avg, path_max])
    channel_attention = keras.activations.sigmoid(channel_attention)
    
    # Apply channel attention weights to the original features
    channel_weighted = Multiply()([conv1, channel_attention])
    
    # Separate average and max pooling to extract spatial features
    spatial_avg = tf.reduce_mean(channel_weighted, axis=[1, 2], keepdims=True)
    spatial_max = tf.reduce_max(channel_weighted, axis=[1, 2], keepdims=True)
    
    # Concatenate along the channel dimension
    spatial_features = Concatenate(axis=-1)([spatial_avg, spatial_max])
    
    # Flatten the spatial features
    flattened_spatial = Flatten()(spatial_features)
    
    # Combine channel and spatial features
    combined_features = Multiply()([channel_weighted, flattened_spatial])
    
    # Fully connected layer for final output
    output_layer = Dense(units=10, activation='softmax')(combined_features)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model