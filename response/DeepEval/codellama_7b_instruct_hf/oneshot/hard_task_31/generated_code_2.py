import keras
from keras.layers import Input, Conv2D, Dropout, Lambda, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main block
    main_block = Conv2D(32, (3, 3), activation='relu')(input_shape)
    main_block = Dropout(0.25)(main_block)
    main_block = Conv2D(64, (3, 3), activation='relu')(main_block)

    # Define the branch path
    branch_path = Conv2D(16, (1, 1), activation='relu')(input_shape)

    # Add the two paths
    added_path = keras.layers.add([main_block, branch_path])

    # Define the second block
    second_block = Lambda(lambda x: tf.split(x, 3, axis=-1))(added_path)
    second_block = Conv2D(16, (1, 1), activation='relu')(second_block[0])
    second_block = Dropout(0.25)(second_block)
    second_block = Conv2D(32, (3, 3), activation='relu')(second_block)
    second_block = Dropout(0.25)(second_block)
    second_block = Conv2D(64, (5, 5), activation='relu')(second_block)

    # Concatenate the outputs from the two blocks
    concatenated = keras.layers.concatenate([second_block, added_path])

    # Flatten the output and add a fully connected layer
    flattened = Flatten()(concatenated)
    output_layer = Dense(10, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=input_shape, outputs=output_layer)

    return model