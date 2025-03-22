import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    # Split the input into three groups along the last dimension
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    group1 = split_inputs[0]  # First group remains unchanged
    group2 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])  # Second group with conv
    group3 = split_inputs[2]  # Third group remains unchanged

    # Combine group2 and group3
    combined_group = layers.Concatenate()([group2, group3])
    main_path_output = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(combined_group)

    # Branch path
    branch_path_output = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse the outputs of both paths
    fused_output = layers.Add()([main_path_output, branch_path_output])

    # Flatten the combined output
    flattened_output = layers.Flatten()(fused_output)
    
    # Fully connected layer for classification
    dense_output = layers.Dense(units=10, activation='softmax')(flattened_output)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=dense_output)

    return model