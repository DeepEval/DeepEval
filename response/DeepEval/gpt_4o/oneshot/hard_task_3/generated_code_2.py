import keras
from keras.layers import Input, Conv2D, Dropout, Lambda, Concatenate, Add, Dense, Flatten
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split_input)(input_layer)

    def process_group(x):
        x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = Dropout(rate=0.3)(x)
        return x

    # Process each split group with the defined sequence
    group1 = process_group(split_layer[0])
    group2 = process_group(split_layer[1])
    group3 = process_group(split_layer[2])

    # Concatenate the outputs from the three groups
    main_pathway = Concatenate()([group1, group2, group3])

    # Parallel branch pathway with 1x1 convolution
    branch_pathway = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine main and branch pathways using an addition operation
    combined_output = Add()([main_pathway, branch_pathway])

    # Flatten and pass through a fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model