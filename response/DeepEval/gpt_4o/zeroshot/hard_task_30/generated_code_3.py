import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Dual-path structure
    # Main Path
    main_path = Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(128, (3, 3), padding='same', activation='relu')(main_path)
    main_path = Conv2D(3, (3, 3), padding='same', activation='relu')(main_path)  # Restoring to input channels

    # Branch Path
    branch_path = input_layer

    # Combine paths
    combined = Add()([main_path, branch_path])

    # Second Block: Split and extract features with depthwise separable convolutions
    def split_and_extract(x):
        # Splitting into three groups along the channels
        split1, split2, split3 = tf.split(x, num_or_size_splits=3, axis=3)

        # Depthwise separable convolutions with different kernel sizes
        path1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split1)
        path2 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split2)
        path3 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split3)

        # Concatenate the outputs
        return Concatenate(axis=3)([path1, path2, path3])

    # Apply the split and extract logic using Lambda layer
    second_block_output = Lambda(split_and_extract)(combined)

    # Fully connected layers
    flattened = Flatten()(second_block_output)
    fc1 = Dense(256, activation='relu')(flattened)
    output_layer = Dense(10, activation='softmax')(fc1)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()