import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Two parallel paths for channel-wise feature extraction
    path_global_avg_pool = GlobalAveragePooling2D()(conv1)
    path_global_avg_pool = Dense(128, activation='relu')(path_global_avg_pool)
    path_global_avg_pool = Dense(64, activation='relu')(path_global_avg_pool)
    
    path_global_max_pool = GlobalMaxPooling2D()(conv1)
    path_global_max_pool = Dense(128, activation='relu')(path_global_max_pool)
    path_global_max_pool = Dense(64, activation='relu')(path_global_max_pool)
    
    # Add the outputs from the two paths
    added_channels = Add()([path_global_avg_pool, path_global_max_pool])
    
    # Generate channel attention weights
    activation_weights = keras.activations.sigmoid(added_channels)
    
    # Apply channel attention weights to the original features
    conv1_weighted = Multiply()([conv1, keras.backend.expand_dims(activation_weights, axis=-1)])
    
    # Separate average and max pooling to extract spatial features
    avg_pool = GlobalAveragePooling2D()(conv1_weighted)
    max_pool = GlobalMaxPooling2D()(conv1_weighted)
    
    # Concatenate the spatial features along the channel dimension
    spatial_features = Concatenate()([avg_pool, max_pool])
    
    # Flatten the fused feature map
    flattened = Flatten()(spatial_features)
    
    # Fully connected layer for final output
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])