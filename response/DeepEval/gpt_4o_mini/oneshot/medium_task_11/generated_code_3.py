import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dense, Multiply, Concatenate, Flatten, Activation
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Parallel paths
    # Path 1: Global Average Pooling followed by two fully connected layers
    avg_pool = GlobalAveragePooling2D()(conv)
    dense1_avg = Dense(units=128, activation='relu')(avg_pool)
    dense2_avg = Dense(units=64, activation='relu')(dense1_avg)

    # Path 2: Global Max Pooling followed by two fully connected layers
    max_pool = GlobalMaxPooling2D()(conv)
    dense1_max = Dense(units=128, activation='relu')(max_pool)
    dense2_max = Dense(units=64, activation='relu')(dense1_max)

    # Combining the two paths
    combined = keras.layers.Add()([dense2_avg, dense2_max])
    channel_attention = Activation('sigmoid')(combined)
    
    # Applying channel attention to original features
    channel_attention = Dense(units=32, activation='sigmoid')(channel_attention)
    channel_attention = Multiply()([conv, channel_attention])

    # Spatial features extraction
    avg_pool_final = GlobalAveragePooling2D()(channel_attention)
    max_pool_final = GlobalMaxPooling2D()(channel_attention)
    
    # Concatenating spatial features
    spatial_features = Concatenate()([avg_pool_final, max_pool_final])

    # Flattening the concatenated output
    flattened = Flatten()(spatial_features)

    # Fully connected layer for the final output
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model