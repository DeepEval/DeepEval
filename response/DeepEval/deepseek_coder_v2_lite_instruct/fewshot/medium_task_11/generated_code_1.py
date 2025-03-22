import keras
from keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, Dense, Add, Activation, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First path: Global Average Pooling followed by two fully connected layers
    gap1 = AveragePooling2D(pool_size=(8, 8))(conv1)
    gap1_flat = Flatten()(gap1)
    dense1 = Dense(units=64, activation='relu')(gap1_flat)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Second path: Global Max Pooling followed by two fully connected layers
    gmp1 = MaxPooling2D(pool_size=(8, 8))(conv1)
    gmp1_flat = Flatten()(gmp1)
    dense3 = Dense(units=64, activation='relu')(gmp1_flat)
    dense4 = Dense(units=32, activation='relu')(dense3)

    # Merge the outputs from the two paths
    added = Add()([dense2, dense4])
    activation = Activation('sigmoid')(added)

    # Generate channel attention weights
    attention_weights = Multiply()([conv1, activation])

    # Separate average and max pooling to extract spatial features
    avg_pool = AveragePooling2D(pool_size=(8, 8))(attention_weights)
    max_pool = MaxPooling2D(pool_size=(8, 8))(attention_weights)

    # Concatenate spatial features along the channel dimension
    spatial_features = keras.layers.concatenate([avg_pool, max_pool], axis=-1)

    # Combine spatial and channel features
    combined_features = Multiply()([attention_weights, spatial_features])

    # Flatten the combined features
    flattened = Flatten()(combined_features)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model