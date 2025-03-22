import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = Input(shape=input_shape)

    # Main path
    # Split the input tensor into three groups along the last dimension
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # First group remains unchanged
    group1 = split_inputs[0]

    # Second group undergoes feature extraction with a 3x3 convolution
    group2 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_inputs[1])

    # Combine group2 with the third group
    combined_group = Add()([group2, split_inputs[2]])

    # Additional 3x3 convolution on the combined output
    combined_group = Conv2D(32, (3, 3), padding='same', activation='relu')(combined_group)

    # Concatenate the outputs of all three groups
    main_path_output = Concatenate()([group1, combined_group, split_inputs[2]])

    # Branch path using a 1x1 convolutional layer
    branch_path_output = Conv2D(16, (1, 1), padding='same', activation='relu')(inputs)

    # Fuse main and branch paths together using addition
    fused_output = Add()([main_path_output, branch_path_output])

    # Flatten the combined output
    flattened_output = Flatten()(fused_output)

    # Fully connected layer for classification
    classification_output = Dense(10, activation='softmax')(flattened_output)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=inputs, outputs=classification_output)

    return model