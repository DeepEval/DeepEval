import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, BatchNormalization, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate outputs of the main path
    main_concat = Concatenate()([main_path1, main_path2, main_path3])

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add outputs of main and branch paths
    added_features = Add()([main_concat, branch_path])

    # Batch normalization
    batch_norm = BatchNormalization()(added_features)

    # Flatten the result
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model