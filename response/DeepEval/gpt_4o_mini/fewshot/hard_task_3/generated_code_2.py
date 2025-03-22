import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Add, Concatenate, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Splitting input into three groups
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main pathway processing
    def main_path(group):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(group)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        dropout = Dropout(rate=0.25)(conv2)
        return dropout

    # Process each group through the main pathway
    processed_groups = [main_path(group) for group in inputs_groups]
    main_output = Concatenate()(processed_groups)

    # Branch pathway processing
    branch_output = Conv2D(filters=192, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs from both pathways
    combined_output = Add()([main_output, branch_output])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model