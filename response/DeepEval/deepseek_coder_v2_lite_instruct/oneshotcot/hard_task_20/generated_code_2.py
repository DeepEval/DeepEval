import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path_5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)

    # Split the main path into three groups
    split_1x1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(main_path_1x1)
    split_3x3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(main_path_3x3)
    split_5x5 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(main_path_5x5)

    # Concatenate the outputs of the three groups
    concatenated_main = Concatenate(axis=3)([split_1x1, split_3x3, split_5x5])

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Add the branch path to the main path
    fused_features = tf.add(concatenated_main, branch_path)

    # Flatten the fused features
    flattened = Flatten()(fused_features)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model