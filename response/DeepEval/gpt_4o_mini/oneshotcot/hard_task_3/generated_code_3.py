import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Concatenate, Add, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Split input into three groups along the channel dimension
    split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define a function to process each group
    def process_group(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        return Dropout(rate=0.25)(conv2)  # Dropout layer for feature selection

    # Process each split group
    processed_groups = [process_group(group) for group in split_groups]
    
    # Concatenate the outputs from the three groups
    main_pathway = Concatenate()(processed_groups)

    # Branch pathway processing the input
    branch_pathway = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs from both pathways
    combined = Add()([main_pathway, branch_pathway])

    # Flatten the combined output and add a fully connected layer
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model