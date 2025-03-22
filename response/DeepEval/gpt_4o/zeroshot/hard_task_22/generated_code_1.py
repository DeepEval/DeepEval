import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Conv2D, Add, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path: Split input into 3 groups along the channel axis
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Multi-scale feature extraction with separable convolutions
    conv_1x1 = SeparableConv2D(32, (1, 1), activation='relu', padding='same')(split_channels[0])
    conv_3x3 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(split_channels[1])
    conv_5x5 = SeparableConv2D(32, (5, 5), activation='relu', padding='same')(split_channels[2])

    # Concatenate the outputs of the three convolutional layers
    main_path_output = Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5])

    # Branch path: Apply a 1x1 convolution to the input
    branch_path = Conv2D(96, (1, 1), activation='relu', padding='same')(input_layer)  # 32 * 3 = 96 channels

    # Fuse the outputs from main path and branch path
    fused_output = Add()([main_path_output, branch_path])

    # Flatten the combined output
    flattened_output = Flatten()(fused_output)

    # Fully connected layers for classification
    dense_1 = Dense(128, activation='relu')(flattened_output)
    dense_2 = Dense(10, activation='softmax')(dense_1)

    # Define the model
    model = Model(inputs=input_layer, outputs=dense_2)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of creating the model
model = dl_model()
model.summary()