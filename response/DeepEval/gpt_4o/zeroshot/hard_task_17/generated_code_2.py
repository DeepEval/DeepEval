import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, MaxPooling2D, Add, Flatten
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Block 1: Global Average Pooling and fully connected layers for generating weights
    gap = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(32, activation='relu')(gap)  # Adjust number of units as needed
    dense2 = Dense(32, activation='relu')(dense1)  # Should match the input channels

    # Reshape to match input and multiply to get weighted features
    reshaped_weights = Reshape((1, 1, 32))(dense2)  # Reshape to match the input shape
    weighted_features = Multiply()([input_layer, reshaped_weights])

    # Block 2: Two convolutional layers followed by max pooling
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(weighted_features)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Fusing the branch (direct connection from Block 1 output) with Block 2 output
    fused = Add()([weighted_features, max_pool])

    # Final classification block
    flat = Flatten()(fused)
    fc1 = Dense(128, activation='relu')(flat)
    output_layer = Dense(10, activation='softmax')(fc1)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.summary()  # Print the model summary to verify the architecture