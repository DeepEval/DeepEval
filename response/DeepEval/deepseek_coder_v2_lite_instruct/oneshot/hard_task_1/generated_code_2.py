import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, Multiply, Activation, BatchNormalization, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = BatchNormalization()(conv1)

    # Path 1: Global Average Pooling followed by two fully connected layers
    gap = GlobalAveragePooling2D()(conv1)
    fc1 = Dense(128, activation='relu')(gap)
    fc2 = Dense(64, activation='relu')(fc1)

    # Path 2: Global Max Pooling followed by two fully connected layers
    gmp = GlobalMaxPooling2D()(conv1)
    fc3 = Dense(128, activation='relu')(gmp)
    fc4 = Dense(64, activation='relu')(fc3)

    # Concatenate the outputs from both paths and generate channel attention weights
    concat = Concatenate()([fc2, fc4])
    attention_weights = Dense(conv1.shape[3], activation='sigmoid')(concat)

    # Apply channel attention weights to the original features
    channel_attention = Multiply()([conv1, attention_weights])

    # Block 2: Extract spatial features
    avg_pool = GlobalAveragePooling2D()(channel_attention)
    max_pool = GlobalMaxPooling2D()(channel_attention)
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_features = Conv2D(filters=32, kernel_size=(1, 1), activation='sigmoid')(spatial_features)

    # Normalize the spatial features and multiply with channel attention features
    normalized_features = Multiply()([channel_attention, spatial_features])

    # Additional branch to ensure output channels align with the input channels
    branch = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(normalized_features)
    main_path = keras.add([normalized_features, branch])
    main_path = Activation('relu')(main_path)

    # Flatten the features for the final fully connected layer
    flatten_layer = Flatten()(main_path)

    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
# model = dl_model()
# model.summary()