import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Add, Activation, Flatten, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels (RGB)
    
    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Parallel paths
    avg_pool = GlobalAveragePooling2D()(conv)
    avg_dense1 = Dense(units=128, activation='relu')(avg_pool)
    avg_dense2 = Dense(units=64, activation='relu')(avg_dense1)
    
    max_pool = GlobalMaxPooling2D()(conv)
    max_dense1 = Dense(units=128, activation='relu')(max_pool)
    max_dense2 = Dense(units=64, activation='relu')(max_dense1)
    
    # Combining channel features
    avg_output = Dense(units=32, activation='sigmoid')(avg_dense2)
    max_output = Dense(units=32, activation='sigmoid')(max_dense2)

    channel_attention = Add()([avg_output, max_output])
    channel_attention = Activation('sigmoid')(channel_attention)
    
    # Applying channel attention to original features
    channel_features = Multiply()([conv, channel_attention])

    # Spatial feature extraction
    avg_spatial = GlobalAveragePooling2D()(channel_features)
    max_spatial = GlobalMaxPooling2D()(channel_features)
    
    # Concatenating spatial features
    spatial_fused = Concatenate()([avg_spatial, max_spatial])

    # Final fully connected layer
    flatten_layer = Flatten()(spatial_fused)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model