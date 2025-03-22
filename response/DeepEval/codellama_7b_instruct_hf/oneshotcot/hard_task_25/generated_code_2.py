import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First branch
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch1)

    # Second branch
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch2)

    # Third branch
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch3)

    # Upsampling
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Concatenate branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3])

    # Final convolutional layer
    final_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_branches)

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(final_conv)
    flatten = Flatten()(batch_norm)

    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model