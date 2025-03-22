import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Activation, Multiply, Concatenate, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)

    # First parallel path: Global Average Pooling followed by two fully connected layers
    avg_pool = GlobalAveragePooling2D()(x)
    fc1_avg = Dense(64, activation='relu')(avg_pool)
    fc2_avg = Dense(128, activation='relu')(fc1_avg)

    # Second parallel path: Global Max Pooling followed by two fully connected layers
    max_pool = GlobalMaxPooling2D()(x)
    fc1_max = Dense(64, activation='relu')(max_pool)
    fc2_max = Dense(128, activation='relu')(fc1_max)

    # Add the outputs of the two paths
    added = Add()([fc2_avg, fc2_max])

    # Activation function to generate channel attention weights
    activation = Activation('sigmoid')(added)

    # Apply channel attention weights to the original features
    scaled = Multiply()([x, activation])

    # Separate average and max pooling operations to extract spatial features
    avg_pool_spatial = GlobalAveragePooling2D()(scaled)
    max_pool_spatial = GlobalMaxPooling2D()(scaled)

    # Concatenate along the channel dimension
    spatial_features = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])

    # Combine spatial features with channel features
    combined = Multiply()([scaled, spatial_features])

    # Flatten the combined features
    flattened = Dense(128, activation='relu')(combined)

    # Final fully connected layer
    outputs = Dense(10, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()