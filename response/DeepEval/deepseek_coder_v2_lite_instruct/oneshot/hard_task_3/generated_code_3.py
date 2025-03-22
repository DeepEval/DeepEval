import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_1 = tf.split(input_layer, num_or_size_splits=3, axis=3)
    split_2 = tf.split(input_layer, num_or_size_splits=3, axis=3)
    split_3 = tf.split(input_layer, num_or_size_splits=3, axis=3)

    def process_group(group):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1x1)
        dropout = Dropout(0.5)(conv3x3)
        return dropout

    # Process each group
    processed_1 = process_group(split_1[0])
    processed_2 = process_group(split_2[1])
    processed_3 = process_group(split_3[2])

    # Concatenate the outputs from the three groups
    main_pathway = Concatenate()([processed_1, processed_2, processed_3])

    # Parallel branch processing
    branch_pathway = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Add the outputs from both pathways
    combined_output = tf.add(main_pathway, branch_pathway)

    # Flatten the result
    flatten_layer = Flatten()(combined_output)

    # Fully connected layer
    dense_layer = Dense(units=100, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model