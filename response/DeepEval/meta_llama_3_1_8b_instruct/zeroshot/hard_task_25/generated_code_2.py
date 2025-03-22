# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, concatenate, Concatenate, Dense, Add, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)

    # Define inputs
    inputs = Input(shape=input_shape)

    # Main path
    main_path = Conv2D(64, kernel_size=(1, 1), padding='same')(inputs)

    # Split main path into three branches
    left_branch = main_path
    middle_branch = Conv2D(64, kernel_size=(3, 3), padding='same')(main_path)
    right_branch = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(main_path)

    # Process middle branch through 3x3 convolutional layer
    right_branch = Conv2D(64, kernel_size=(3, 3), padding='same')(right_branch)

    # Upsample right branch
    right_branch = UpSampling2D(size=(2, 2))(right_branch)

    # Concatenate outputs of all branches
    merged_branches = concatenate([left_branch, middle_branch, right_branch], axis=3)

    # Apply 1x1 convolutional layer to form main path output
    main_path_output = Conv2D(64, kernel_size=(1, 1), padding='same')(merged_branches)

    # Branch path
    branch_path = Conv2D(64, kernel_size=(1, 1), padding='same')(inputs)

    # Fuse main path and branch path outputs together through addition
    fused_outputs = Add()([main_path_output, branch_path])

    # Flatten output
    flattened_output = tf.keras.layers.Flatten()(fused_outputs)

    # Apply fully connected layer for 10-class classification
    outputs = Dense(10, activation='softmax')(flattened_output)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model