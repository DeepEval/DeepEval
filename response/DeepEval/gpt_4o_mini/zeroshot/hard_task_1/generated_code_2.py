import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, Multiply, Concatenate, Activation, AveragePooling2D, MaxPooling2D, Add
from tensorflow.keras.models import Model

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    # Input layer
    inputs = Input(shape=input_shape)

    # Initial convolutional layer
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)

    # Block 1: Parallel paths for channel attention
    # Path 1: Global Average Pooling followed by two fully connected layers
    path1 = GlobalAveragePooling2D()(x)
    path1 = Dense(32, activation='relu')(path1)
    path1 = Dense(32, activation='relu')(path1)

    # Path 2: Global Max Pooling followed by two fully connected layers
    path2 = GlobalMaxPooling2D()(x)
    path2 = Dense(32, activation='relu')(path2)
    path2 = Dense(32, activation='relu')(path2)

    # Combine paths and generate channel attention weights
    combined = Add()([path1, path2])
    attention_weights = Activation('sigmoid')(combined)

    # Apply attention weights to original features
    channel_attention = Multiply()([x, tf.expand_dims(attention_weights, axis=1)])

    # Block 2: Extract spatial features
    avg_pool = AveragePooling2D(pool_size=(2, 2))(x)
    max_pool = MaxPooling2D(pool_size=(2, 2))(x)

    # Concatenate along channel dimension
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])

    # 1x1 convolution and sigmoid activation
    spatial_features = Conv2D(32, (1, 1), activation='sigmoid')(spatial_features)

    # Multiply with channel dimension features from Block 1
    spatial_attention = Multiply()([channel_attention, spatial_features])

    # Additional branch with a 1x1 convolution to align output channels
    output_branch = Conv2D(32, (1, 1))(spatial_attention)

    # Add to the main path and activate
    x = Add()([channel_attention, output_branch])
    x = Activation('relu')(x)

    # Final classification through a fully connected layer
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()