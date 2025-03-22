import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, Lambda, SeparableConv2D, Concatenate, Dense, Flatten
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Dual-path structure
    # Main Path
    main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = Conv2D(3, (3, 3), activation='relu', padding='same')(main_path)  # Restore channel dimension

    # Branch Path (Direct connection)
    branch_path = input_layer

    # Combine paths through addition
    combined = Add()([main_path, branch_path])

    # Second Block: Split and process with depthwise separable convolutions
    # Split input into three parts along the channel dimension
    split_parts = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(combined)

    # Extract features with different kernel sizes
    processed_1x1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_parts[0])
    processed_3x3 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_parts[1])
    processed_5x5 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_parts[2])

    # Concatenate processed outputs
    concatenated = Concatenate()([processed_1x1, processed_3x3, processed_5x5])

    # Fully connected layers
    flatten = Flatten()(concatenated)
    dense1 = Dense(128, activation='relu')(flatten)
    output_layer = Dense(10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
model = dl_model()
model.summary()