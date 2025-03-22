import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    main_path_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(main_path[0])
    main_path_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path[1])
    main_path_5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(main_path[2])
    main_output = Concatenate(axis=3)([main_path_1x1, main_path_3x3, main_path_5x5])

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Add outputs from main and branch paths
    added_output = Add()([main_output, branch_path])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(added_output)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model