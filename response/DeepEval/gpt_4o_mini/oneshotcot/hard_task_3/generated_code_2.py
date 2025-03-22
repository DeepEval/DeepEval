import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Concatenate, Add, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Splitting the input into three groups along the channel dimension
    splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    def feature_extraction_branch(input_tensor):
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Dropout(0.25)(x)  # Dropout layer to mitigate overfitting
        return x

    # Process each split through the feature extraction branch
    branch_outputs = [feature_extraction_branch(split) for split in splits]

    # Concatenate outputs of the three branches
    main_pathway = Concatenate()(branch_outputs)

    # Branch pathway processes the input through a 1x1 convolution
    branch_pathway = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the main pathway and branch pathway using an addition operation
    combined_output = Add()([main_pathway, branch_pathway])

    # Flatten the result and add the fully connected layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model