import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    # Main path: Conv + Dropout + Conv
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Dropout(0.2)(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(main_path)

    # Branch path: direct connection from input
    branch_path = input_layer

    # Add the outputs from both paths
    block_output = Add()([main_path, branch_path])

    # Second Block
    # Split the input into three groups
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(block_output)

    # Separable Convolutions with Dropout for each group
    sep_conv_outputs = []
    for i, kernel_size in enumerate([(1, 1), (3, 3), (5, 5)]):
        sep_conv = SeparableConv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(split_tensors[i])
        sep_conv = Dropout(0.2)(sep_conv)
        sep_conv_outputs.append(sep_conv)

    # Concatenate outputs from the three separable convolutions
    concatenated_output = Concatenate()(sep_conv_outputs)

    # Flatten the result
    flatten_layer = Flatten()(concatenated_output)

    # Fully connected layer for final predictions
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model