import keras
from keras.layers import Input, Conv2D, Dropout, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define a function to process each group
    def process_group(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        return Dropout(0.25)(conv2)  # Add dropout after convolutions

    # Process each of the split groups
    group_outputs = [process_group(group) for group in split_tensors]

    # Concatenate the outputs from the three groups
    main_pathway = Concatenate()(group_outputs)

    # Parallel branch pathway with a 1x1 convolution
    branch_pathway = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs of the main and branch pathways
    combined_output = Add()([main_pathway, branch_pathway])

    # Flatten the output and add a fully connected layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model