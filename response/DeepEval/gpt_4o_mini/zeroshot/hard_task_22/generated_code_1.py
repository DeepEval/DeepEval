import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Split the input tensor into three groups along the channel axis
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

    # Main path: multi-scale feature extraction with separable convolutional layers
    conv1 = layers.SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    conv2 = layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    conv3 = layers.SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])

    # Concatenate the outputs from the three branches
    main_path_output = layers.Concatenate()([conv1, conv2, conv3])

    # Branch path: 1x1 convolution to align the number of output channels
    branch_output = layers.Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

    # Fusion through addition
    fused_output = layers.Add()([main_path_output, branch_output])

    # Flatten the output
    flatten_output = layers.Flatten()(fused_output)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flatten_output)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    
    # Output layer for 10-class classification
    output_layer = layers.Dense(10, activation='softmax')(dense2)

    # Constructing the model
    model = models.Model(inputs=input_tensor, outputs=output_layer)

    return model