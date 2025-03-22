import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First block: Parallel branches
    branch1x1 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    branch5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(inputs)

    # Concatenate branches
    combined = Concatenate(axis=-1)([branch1x1, branch3x3, branch5x5, branch_pool])

    # Second block: Global Average Pooling and Fully Connected Layers
    gap = GlobalAveragePooling2D()(combined)
    dense1 = Dense(128, activation='relu')(gap)
    dense2 = Dense(128, activation='relu')(dense1)

    # Reshape and Multiply
    reshape_weights = Reshape((1, 1, 128))(dense2)
    multiplied_feature_map = Multiply()([inputs, reshape_weights])

    # Final Fully Connected Layer
    outputs = Dense(10, activation='softmax')(multiplied_feature_map)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()