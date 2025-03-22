import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply, add

def dl_model():     
    
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Feature extraction using parallel paths
    path1 = GlobalAveragePooling2D()(conv)
    path1 = Dense(units=128, activation='relu')(path1)
    path1 = Dense(units=1, activation='sigmoid')(path1)

    path2 = GlobalMaxPooling2D()(conv)
    path2 = Dense(units=128, activation='relu')(path2)
    path2 = Dense(units=1, activation='sigmoid')(path2)

    # Channel attention weights
    ca = Multiply()([path1, path2])
    ca = Flatten()(ca)
    ca = Dense(units=32, activation='relu')(ca)
    ca = Dense(units=32, activation='sigmoid')(ca)
    ca = Reshape((1, 1, 32))(ca)

    # Element-wise multiplication to enhance features
    enhanced_features = Multiply()([conv, ca])

    # Spatial feature extraction
    avg_pool = AveragePooling2D()(enhanced_features)
    max_pool = MaxPooling2D()(enhanced_features)

    # Concatenation of spatial features
    spatial_features = Concatenate()([avg_pool, max_pool])

    # Fusion of channel and spatial features
    fused_features = Multiply()([spatial_features, enhanced_features])

    # Final fully connected layer
    flatten_layer = Flatten()(fused_features)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model