import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    # Step 2: Add convolutional layers with depthwise separable convolutions
    main_path_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_layer)
    main_path_3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
    main_path_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu')(input_layer)

    # Step 3: Concatenate outputs from the main path
    main_output = Concatenate()([main_path_1x1, main_path_3x3, main_path_5x5])

    # Branch path
    # Step 4: Add 1x1 convolutional layer in the branch path
    branch_path = Conv2D(64, (1, 1), padding='same', activation='relu')(input_layer)

    # Step 5: Add batch normalization
    batch_norm_main = BatchNormalization()(main_output)
    batch_norm_branch = BatchNormalization()(branch_path)

    # Step 6: Add flatten layer
    flatten_main = Flatten()(batch_norm_main)
    flatten_branch = Flatten()(batch_norm_branch)

    # Step 7: Add dense layers
    dense1 = Dense(128, activation='relu')(flatten_main)
    dense2 = Dense(128, activation='relu')(flatten_branch)

    # Step 8: Add output layer
    add_layer = tf.add(dense1, dense2)
    output_layer = Dense(10, activation='softmax')(add_layer)

    # Step 9: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model