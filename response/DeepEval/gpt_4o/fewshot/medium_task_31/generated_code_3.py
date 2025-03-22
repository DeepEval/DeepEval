import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Function to process each group with different convolutional kernels
    def process_group(input_tensor):
        group1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor[0])
        group2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor[1])
        group3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor[2])
        return [group1, group2, group3]

    # Split the input into 3 groups along the channel dimension
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Process each group separately and concatenate the results
    processed_groups = process_group(split_channels)
    concatenated_features = Concatenate()(processed_groups)

    # Flatten the concatenated features
    flatten_layer = Flatten()(concatenated_features)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model