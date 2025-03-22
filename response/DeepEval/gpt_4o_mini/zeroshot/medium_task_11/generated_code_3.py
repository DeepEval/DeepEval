import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, Multiply, Flatten, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    # Path 1: Global Average Pooling followed by Fully Connected Layers
    avg_pool = GlobalAveragePooling2D()(x)
    avg_fc1 = Dense(64, activation='relu')(avg_pool)
    avg_fc2 = Dense(32, activation='relu')(avg_fc1)

    # Path 2: Global Max Pooling followed by Fully Connected Layers
    max_pool = GlobalMaxPooling2D()(x)
    max_fc1 = Dense(64, activation='relu')(max_pool)
    max_fc2 = Dense(32, activation='relu')(max_fc1)

    # Channel attention weights
    channel_attention = Add()([avg_fc2, max_fc2])
    channel_attention = Dense(32, activation='sigmoid')(channel_attention)

    # Apply attention weights to the original features
    channel_features = Multiply()([x, tf.expand_dims(tf.expand_dims(channel_attention, axis=1), axis=1)])

    # Spatial feature extraction
    spatial_avg = GlobalAveragePooling2D()(channel_features)
    spatial_max = GlobalMaxPooling2D()(channel_features)

    # Concatenate spatial features
    spatial_features = Concatenate()([spatial_avg, spatial_max])

    # Combine spatial features with channel features
    combined_features = Multiply()([spatial_features, tf.expand_dims(tf.expand_dims(channel_attention, axis=1), axis=1)])

    # Flatten and fully connected layer for output
    flattened = Flatten()(combined_features)
    output = Dense(10, activation='softmax')(flattened)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=output)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # To display the model architecture