import keras
from keras.layers import Input, Conv2D, Dropout, Concatenate, Add, Dense, Flatten, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Splitting the input into three groups along the channel dimension
    inputs_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    def process_group(group):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
        dropout = Dropout(rate=0.5)(conv3x3)
        return dropout

    # Processing each group
    processed_group1 = process_group(inputs_groups[0])
    processed_group2 = process_group(inputs_groups[1])
    processed_group3 = process_group(inputs_groups[2])

    # Concatenate the outputs of the processed groups
    main_path = Concatenate()([processed_group1, processed_group2, processed_group3])

    # Parallel branch path
    branch_path = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine main and branch pathways
    combined = Add()([main_path, branch_path])

    # Fully connected layer for classification
    flatten = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model