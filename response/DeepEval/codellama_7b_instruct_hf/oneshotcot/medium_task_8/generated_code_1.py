import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    # Main path
    main_path = input_layer
    # Split the input into three groups along the last dimension
    main_path = Lambda(lambda x: tf.split(x, 3, axis=-1))(main_path)
    # Apply feature extraction to the second group
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path[1])
    # Combine the output of the second group with the third group
    main_path = Concatenate()([main_path[0], main_path[2]])
    # Apply additional feature extraction to the combined output
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    # Concatenate the outputs of all three groups
    main_path = Concatenate()(main_path)
    # Branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(input_layer)
    # Fuse the outputs from both the main and branch paths through addition
    combined_output = Concatenate()([main_path, branch_path])
    # Flatten the combined output
    flattened_output = Flatten()(combined_output)
    # Pass the flattened output through a fully connected layer
    output = Dense(10, activation='softmax')(flattened_output)
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)
    return model