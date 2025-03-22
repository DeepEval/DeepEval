import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, add, concatenate

def dl_model():

    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # Block 1
    main_path = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(inputs)
    main_path = BatchNormalization()(main_path)
    main_path = Activation('relu')(main_path)
    main_path = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(main_path)
    main_path = BatchNormalization()(main_path)
    main_path = Activation('relu')(main_path)
    main_path = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(main_path)
    main_path = BatchNormalization()(main_path)
    main_path = Activation('relu')(main_path)

    branch_path = Conv2D(filters=256, kernel_size=(1, 1), padding='same')(inputs)
    branch_path = BatchNormalization()(branch_path)
    branch_path = Activation('relu')(branch_path)

    # Concatenate outputs from main and branch paths
    combined_path = add([main_path, branch_path])
    combined_path = Activation('relu')(combined_path)

    # Block 2
    block_2_output = MaxPooling2D(pool_size=(1, 1), strides=1)(combined_path)
    block_2_output = concatenate([block_2_output, MaxPooling2D(pool_size=(2, 2), strides=2)(combined_path)])
    block_2_output = concatenate([block_2_output, MaxPooling2D(pool_size=(4, 4), strides=4)(combined_path)])

    # Block 2 output is flattened and concatenated
    block_2_output = Flatten()(block_2_output)

    # Fully connected layers
    dense_layer_1 = Dense(units=64, activation='relu')(block_2_output)
    dense_layer_2 = Dense(units=10, activation='softmax')(dense_layer_1)

    # Model construction
    model = keras.Model(inputs=inputs, outputs=dense_layer_2)

    return model