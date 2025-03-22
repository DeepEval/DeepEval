import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, SeparableConv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    # Split the input into three groups along the channel axis
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Extract features with different kernel sizes
    conv1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same')(split_layer[0])
    conv3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(split_layer[1])
    conv5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same')(split_layer[2])

    # Concatenate the outputs
    concat_features = Concatenate(axis=3)([conv1x1, conv3x3, conv5x5])

    # Second block
    # First branch: 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(concat_features)

    # Second branch: 1x1 convolution followed by two 3x3 convolutions
    branch2_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(concat_features)
    branch2_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(branch2_1)
    branch2_3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(branch2_2)

    # Third branch: max pooling
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(concat_features)

    # Concatenate outputs from all branches
    second_block_output = Concatenate(axis=3)([branch1, branch2_3, branch3])

    # Global average pooling
    gap_layer = GlobalAveragePooling2D()(second_block_output)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(gap_layer)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model