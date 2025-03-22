import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, Concatenate, AveragePooling2D, MaxPooling2D, BatchNormalization

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Block 1: Convolutional Layer
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)

    # Path 1: Global Average Pooling followed by two fully connected layers
    gap = GlobalAveragePooling2D()(x)
    dense1 = Dense(128, activation='relu')(gap)
    dense2 = Dense(64, activation='relu')(dense1)
    path1_output = Dense(32, activation='sigmoid')(dense2)  # Output shape should match input channels

    # Path 2: Global Max Pooling followed by two fully connected layers
    gmp = GlobalMaxPooling2D()(x)
    dense1_gmp = Dense(128, activation='relu')(gmp)
    dense2_gmp = Dense(64, activation='relu')(dense1_gmp)
    path2_output = Dense(32, activation='sigmoid')(dense2_gmp)  # Output shape should match input channels

    # Element-wise addition of path outputs and activation
    added_paths = Add()([path1_output, path2_output])
    attention_weights = Activation('sigmoid')(added_paths)

    # Apply attention weights to the original features
    x_atten = Multiply()([x, attention_weights])

    # Block 2: Spatial Feature Extraction
    avg_pool = AveragePooling2D((3, 3))(x_atten)
    max_pool = MaxPooling2D((3, 3))(x_atten)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    conv1x1 = Conv2D(32, (1, 1), activation='sigmoid')(concat)

    # Normalize features and multiply with Block 1 output
    normalized_features = Multiply()([x_atten, conv1x1])

    # Additional branch to ensure output channels align with input channels
    final_branch = Conv2D(32, (1, 1))(normalized_features)
    output = Add()([normalized_features, final_branch])
    output = Activation('relu')(output)

    # Final classification
    flatten = tf.keras.layers.Flatten()(output)
    dense_final = Dense(10, activation='softmax')(flatten)

    # Model
    model = Model(inputs=inputs, outputs=dense_final)

    return model

# Example usage
model = dl_model()
model.summary()