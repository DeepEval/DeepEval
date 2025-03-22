from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1: Primary path
    primary_path = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='valid')(inputs)
    primary_path = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(primary_path)
    primary_path = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='valid')(primary_path)

    # Block 1: Branch path
    branch_path = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(inputs)
    branch_path = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='valid')(branch_path)

    # Concatenate features
    concat_path = layers.concatenate([primary_path, branch_path])

    # Block 2: Reshape and channel shuffling
    shape_info = keras.backend.int_shape(concat_path)
    reshaped_path = layers.Reshape((shape_info[1], shape_info[2], shape_info[3] // 4, 4))(concat_path)
    permutated_path = layers.Permute((1, 2, 4, 3))(reshaped_path)
    shuffled_path = layers.Reshape((shape_info[1], shape_info[2], shape_info[3], 4))(permutated_path)

    # Final fully connected layer
    outputs = layers.Dense(10, activation='softmax')(shuffled_path)

    # Model definition
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model