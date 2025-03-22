import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 64))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)

    # Concatenate the outputs from the main and branch paths
    combined = Add()([main_path, branch_path])

    # Flatten the combined output
    flattened = Flatten()(combined)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model