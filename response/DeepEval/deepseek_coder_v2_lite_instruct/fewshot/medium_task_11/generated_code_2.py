import keras
from keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, Dense, Add, Activation, Multiply, Flatten, Dropout
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First parallel path: Global Average Pooling followed by two fully connected layers
    gap = tf.reduce_mean(conv1, axis=[1, 2])
    dense1 = Dense(units=64, activation='relu')(gap)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Second parallel path: Global Max Pooling followed by two fully connected layers
    gmp = tf.reduce_max(conv1, axis=[1, 2])
    dense3 = Dense(units=64, activation='relu')(gmp)
    dense4 = Dense(units=64, activation='relu')(dense3)

    # Add the outputs from the two paths
    added = Add()([dense2, dense4])
    attention_weights = Activation('sigmoid')(added)

    # Apply attention weights to the original features
    scaled_features = Multiply()([conv1, attention_weights])

    # Extract spatial features using average and max pooling
    avg_pool = tf.reduce_mean(scaled_features, axis=[1, 2])
    max_pool = tf.reduce_max(scaled_features, axis=[1, 2])

    # Concatenate the spatial features along the channel dimension
    spatial_features = tf.stack([avg_pool, max_pool], axis=-1)

    # Combine channel and spatial features
    combined_features = Multiply()([scaled_features, spatial_features])

    # Flatten the combined features
    flattened = Flatten()(combined_features)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()