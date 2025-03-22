import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, Flatten
from keras.models import Model

def dl_model():
    # Block 1
    def block_1(input_tensor):
        # Global Average Pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        # Fully connected layers
        fc1 = Dense(units=32, activation='relu')(gap)
        fc2 = Dense(units=32, activation='relu')(fc1)
        # Reshape to match input shape
        reshaped = keras.layers.Reshape(target_shape=(1, 1, 32))(fc2)
        # Multiply with input to produce weighted feature output
        weighted_features = Multiply()([input_tensor, reshaped])
        return weighted_features

    # Block 2
    def block_2(input_tensor):
        # Two 3x3 convolutional layers
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
        # Max pooling layer
        max_pool = MaxPooling2D(pool_size=(2, 2))(conv2)
        return max_pool

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Apply Block 1
    block1_output = block_1(input_layer)

    # Apply Block 2
    block2_output = block_2(block1_output)

    # Branch from Block 1
    branch_output = block_1(input_layer)

    # Add the main path and the branch outputs
    added_output = Add()([block2_output, branch_output])

    # Flatten the combined output
    flattened_output = Flatten()(added_output)

    # Final classification layers
    fc1 = Dense(units=64, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(fc1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()