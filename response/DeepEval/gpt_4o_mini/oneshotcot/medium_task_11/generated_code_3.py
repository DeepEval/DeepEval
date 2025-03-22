import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Concatenate, Flatten, Activation

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Parallel paths
    # Path 1: Global Average Pooling
    avg_pooling = GlobalAveragePooling2D()(conv)
    dense1_avg = Dense(units=128, activation='relu')(avg_pooling)
    dense2_avg = Dense(units=128, activation='relu')(dense1_avg)

    # Path 2: Global Max Pooling
    max_pooling = GlobalMaxPooling2D()(conv)
    dense1_max = Dense(units=128, activation='relu')(max_pooling)
    dense2_max = Dense(units=128, activation='relu')(dense1_max)

    # Channel attention weights
    channel_features = Concatenate()([dense2_avg, dense2_max])
    attention_weights = Dense(units=32, activation='sigmoid')(channel_features)
    
    # Apply channel attention weights to the original features
    channel_attention = Multiply()([conv, Activation('relu')(attention_weights)])

    # Spatial features extraction
    avg_spatial = GlobalAveragePooling2D()(channel_attention)
    max_spatial = GlobalMaxPooling2D()(channel_attention)

    # Concatenate spatial features
    spatial_features = Concatenate()([avg_spatial, max_spatial])

    # Flatten and feed into a fully connected layer
    flatten = Flatten()(spatial_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model