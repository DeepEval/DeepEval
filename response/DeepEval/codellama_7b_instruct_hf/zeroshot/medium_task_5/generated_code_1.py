from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the main path
    main_path = input_layer
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)

    # Define the branch path
    branch_path = input_layer
    branch_path = Conv2D(16, (3, 3), activation='relu')(branch_path)
    branch_path = MaxPooling2D((2, 2))(branch_path)

    # Combine the main and branch paths
    combined_path = main_path + branch_path

    # Flatten the combined path
    flattened_path = Flatten()(combined_path)

    # Add fully connected layers
    flattened_path = Dense(128, activation='relu')(flattened_path)
    flattened_path = Dense(10, activation='softmax')(flattened_path)

    # Create the model
    model = Model(inputs=input_layer, outputs=flattened_path)

    return model