import keras
from keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def split_into_groups(x):
        return Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Group 1, unchanged
    group1 = split_into_groups(input_layer)[0]
    # Group 2, 3x3 convolutional layer
    group2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group1)
    # Group 3, additional 3x3 convolution
    group3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group2)
    # Concatenate outputs from all three groups
    main_path_output = Concatenate()([group2, group3])

    # Branch path
    branch_layer = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    branch_output = Flatten()(branch_layer)

    # Combine outputs from main and branch paths
    combined_output = Add()([main_path_output, branch_output])

    # Flatten and fully connected layers
    flattened = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Model structure
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model