import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = input_layer
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)
    main_path = Flatten()(main_path)

    # Branch path
    branch_path = input_layer
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch_path)
    branch_path = Flatten()(branch_path)

    # Concatenate main and branch paths
    main_branch_path = Concatenate()([main_path, branch_path])

    # Addition layer
    main_branch_path = Add()([main_branch_path, main_branch_path])

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(main_branch_path)

    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model