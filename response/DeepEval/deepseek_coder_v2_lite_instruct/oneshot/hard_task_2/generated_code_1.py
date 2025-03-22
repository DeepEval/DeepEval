import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Process each group with a series of convolutions
    def process_group(group):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        return path3

    # Apply process_group to each split group
    processed_groups = [process_group(group) for group in split_layer]

    # Combine the processed groups using addition
    combined_path = Add()(processed_groups)

    # Add the combined path back to the original input
    fused_output = Add()([combined_path, input_layer])

    # Flatten the combined features
    flattened_layer = Flatten()(fused_output)

    # Pass the flattened output through a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flattened_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model