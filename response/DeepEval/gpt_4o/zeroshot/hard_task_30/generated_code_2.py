import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, Lambda, DepthwiseConv2D, Concatenate, Dense, Flatten
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block: Dual-path structure
    # Main path
    main_path = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    main_path = Conv2D(64, (3, 3), activation='relu', padding='same')(main_path)
    main_path = Conv2D(3, (3, 3), activation='relu', padding='same')(main_path)

    # Branch path (identity)
    branch_path = input_layer

    # Combining main path and branch path
    combined = Add()([main_path, branch_path])

    # Second block: Splitting and using depthwise separable convolutions
    # Split into three groups along the channel
    split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(combined)

    # Different kernel sizes for depthwise separable convolutions
    group1 = DepthwiseConv2D((1, 1), activation='relu', padding='same')(split_groups[0])
    group2 = DepthwiseConv2D((3, 3), activation='relu', padding='same')(split_groups[1])
    group3 = DepthwiseConv2D((5, 5), activation='relu', padding='same')(split_groups[2])

    # Concatenate the outputs from the three groups
    concatenated = Concatenate()([group1, group2, group3])

    # Fully connected layers
    flattened = Flatten()(concatenated)
    fc1 = Dense(128, activation='relu')(flattened)
    output_layer = Dense(10, activation='softmax')(fc1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model