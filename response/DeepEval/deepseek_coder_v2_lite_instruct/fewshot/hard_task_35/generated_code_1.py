import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def shared_block(input_tensor):
        # Global Average Pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        # Two fully connected layers
        dense1 = Dense(units=64, activation='relu')(gap)
        dense2 = Dense(units=32, activation='relu')(dense1)
        # Reshape to match input shape
        reshaped = Reshape(target_shape=(1, 1, 32))(dense2)
        # Element-wise multiplication with input
        multiplied = Multiply()([input_tensor, reshaped])
        return multiplied

    # Apply shared block to both branches
    branch1 = shared_block(input_tensor=input_layer)
    branch2 = shared_block(input_tensor=input_layer)

    # Concatenate outputs from both branches
    concatenated = Concatenate()([branch1, branch2])
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model