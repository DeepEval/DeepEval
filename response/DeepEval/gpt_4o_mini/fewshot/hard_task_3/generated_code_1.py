import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Concatenate, Add, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Split the input into three groups along the channel dimension
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    def feature_extraction(group):
        # Each group undergoes a sequence of 1x1 and 3x3 convolutions
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(group)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        return Dropout(rate=0.5)(conv2)  # Apply dropout to mitigate overfitting

    # Process each group
    processed_groups = [feature_extraction(group) for group in inputs_groups]

    # Concatenate the processed groups to form the main pathway
    main_path = Concatenate(axis=-1)(processed_groups)

    # Branch pathway to match the output dimension of the main pathway
    branch_path = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs from both pathways
    combined_output = Add()([main_path, branch_path])

    # Flatten the output and apply a fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model