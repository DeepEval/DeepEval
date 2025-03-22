from tensorflow.keras.layers import Input, Lambda, Conv2D, SeparableConv2D, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three along the channel dimension
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main path
    # Apply depthwise separable convolutions with different kernel sizes
    conv_1x1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_channels[0])
    conv_3x3 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_channels[1])
    conv_5x5 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_channels[2])

    # Concatenate the outputs from the three depthwise separable convolutions
    main_path_output = Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5])

    # Branch path
    # 1x1 Convolution to align with the number of output channels
    branch_path_output = Conv2D(96, (1, 1), padding='same', activation='relu')(input_layer)

    # Combine the main and branch paths
    combined_output = Add()([main_path_output, branch_path_output])

    # Flatten the combined output
    flattened_output = Flatten()(combined_output)

    # Fully connected layers for classification
    dense_1 = Dense(128, activation='relu')(flattened_output)
    output_layer = Dense(10, activation='softmax')(dense_1)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model