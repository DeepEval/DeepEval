import keras
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the main path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Flatten()(main_path)
    main_path = Dense(64, activation='relu')(main_path)
    main_path = Dense(64, activation='relu')(main_path)
    main_path = Dense(10, activation='softmax')(main_path)

    # Define the branch path
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_layer)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(64, activation='relu')(branch_path)
    branch_path = Dense(10, activation='softmax')(branch_path)

    # Add the main and branch paths
    output_layer = Concatenate()([main_path, branch_path])

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model