from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))

    # Main path
    main_path = Conv2D(32, (1, 1), activation='relu')(input_layer)
    main_path = Conv2D(32, (1, 1), activation='relu')(main_path)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)

    # Branch path
    branch_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch_path = MaxPooling2D((2, 2))(branch_path)

    # Concatenate main and branch paths
    merged_path = Concatenate()([main_path, branch_path])

    # Flatten and add layers
    flattened = Flatten()(merged_path)
    flattened = Dense(128, activation='relu')(flattened)
    flattened = Dense(10, activation='softmax')(flattened)

    # Create and return model
    model = Model(inputs=input_layer, outputs=flattened)
    return model