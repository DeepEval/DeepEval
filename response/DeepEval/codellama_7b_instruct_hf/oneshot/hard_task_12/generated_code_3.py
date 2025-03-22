import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 64))

    # Main path
    main_path = Conv2D(32, (1, 1), activation='relu')(input_layer)
    main_path = Conv2D(32, (1, 1), activation='relu')(main_path)
    main_path = Concatenate()([main_path, main_path])

    # Branch path
    branch_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch_path = Concatenate()([branch_path, branch_path])

    # Combine the main and branch paths
    output_layer = Add()([main_path, branch_path])

    # Flatten the output
    output_layer = Flatten()(output_layer)

    # Dense layers
    output_layer = Dense(128, activation='relu')(output_layer)
    output_layer = Dense(64, activation='relu')(output_layer)
    output_layer = Dense(10, activation='softmax')(output_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model