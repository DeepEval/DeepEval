import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Concatenate, Add, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def group_processing(input_tensor):
        # Splitting the input into three groups along the channel dimension
        input_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        outputs = []
        for group in input_groups:
            x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(group)
            x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
            x = Dropout(rate=0.5)(x)  # Apply dropout for regularization
            outputs.append(x)

        return outputs

    # Process the input through the main pathway
    main_path_outputs = group_processing(input_tensor=input_layer)
    main_path = Concatenate()(main_path_outputs)

    # Process the input through the branch pathway
    branch_path = Conv2D(filters=32*3, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs from both pathways
    combined_output = Add()([main_path, branch_path])

    # Flatten and add a fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model