import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 64)

    # Define the main path
    main_path = Conv2D(32, (1, 1), activation='relu')(input_shape)
    main_path = Conv2D(64, (1, 1), activation='relu')(main_path)
    main_path = Conv2D(128, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)

    # Define the branch path
    branch_path = Conv2D(128, (3, 3), activation='relu')(input_shape)
    branch_path = MaxPooling2D((2, 2))(branch_path)

    # Combine the main and branch paths using addition
    combined_path = Add()([main_path, branch_path])

    # Flatten the combined path and pass through two fully connected layers for classification
    flattened_path = Flatten()(combined_path)
    output_layer = Dense(10, activation='softmax')(flattened_path)

    # Define the model
    model = keras.Model(inputs=input_shape, outputs=output_layer)

    return model