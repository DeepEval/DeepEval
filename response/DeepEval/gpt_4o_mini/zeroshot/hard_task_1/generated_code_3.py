import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, Add, Multiply, Concatenate, Activation, Reshape
from tensorflow.keras.models import Model

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    # Input layer
    inputs = Input(shape=input_shape)

    # Initial convolutional layer
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)

    # Block 1
    # Path 1: Global Average Pooling and Fully Connected Layers
    path1 = GlobalAveragePooling2D()(x)
    path1 = Dense(64, activation='relu')(path1)
    path1 = Dense(64, activation='relu')(path1)

    # Path 2: Global Max Pooling and Fully Connected Layers
    path2 = GlobalMaxPooling2D()(x)
    path2 = Dense(64, activation='relu')(path2)
    path2 = Dense(64, activation='relu')(path2)

    # Combine the two paths
    combined = Add()([path1, path2])
    combined = Activation('sigmoid')(combined)

    # Reshape to match the channel dimensions
    combined = Reshape((1, 1, 64))(combined)

    # Channel attention
    attention = Multiply()([x, combined])

    # Block 2: Spatial Feature Extraction
    avg_pool = GlobalAveragePooling2D()(attention)
    max_pool = GlobalMaxPooling2D()(attention)

    # Concatenate along the channel dimension
    concat = Concatenate()([avg_pool, max_pool])

    # Apply a 1x1 convolution and sigmoid activation
    spatial_features = Conv2D(64, (1, 1), padding='same')(Reshape((1, 1, 128))(concat))
    spatial_features = Activation('sigmoid')(spatial_features)

    # Element-wise multiplication of spatial features with channel attention features
    spatial_attention = Multiply()([attention, spatial_features])

    # Final 1x1 convolution to ensure output channels align with input channels
    final_conv = Conv2D(64, (1, 1), padding='same')(spatial_attention)

    # Add residual connection
    added = Add()([final_conv, x])
    activated_output = Activation('relu')(added)

    # Classification layer
    x = GlobalAveragePooling2D()(activated_output)
    x = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs, x)

    return model

# Create the model
model = dl_model()
model.summary()