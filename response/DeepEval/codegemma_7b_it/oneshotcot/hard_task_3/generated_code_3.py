import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dropout, Dense, Flatten

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Main pathway
    main_pathway = []
    for group in split_input:
        # Extract deep features
        group = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(group)
        group = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group)

        # Dropout for feature selection
        group = Dropout(0.25)(group)

        main_pathway.append(group)

    # Concatenate the outputs from the three groups
    main_pathway = Concatenate()(main_pathway)

    # Branch pathway
    branch_pathway = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs from both pathways
    combined_output = Add()([main_pathway, branch_pathway])

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(combined_output)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model