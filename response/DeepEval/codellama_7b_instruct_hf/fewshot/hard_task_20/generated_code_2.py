import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the main path
    main_path = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    main_path = Concatenate()(main_path)
    main_path = Conv2D(64, (1, 1), activation='relu')(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = Conv2D(64, (5, 5), activation='relu')(main_path)
    main_path = Flatten()(main_path)

    # Define the branch path
    branch_path = Conv2D(64, (1, 1), activation='relu')(input_layer)
    branch_path = MaxPooling2D((2, 2))(branch_path)

    # Combine the main and branch paths
    output_layer = Concatenate()([main_path, branch_path])

    # Add two fully connected layers for classification
    output_layer = Dense(64, activation='relu')(output_layer)
    output_layer = Dense(10, activation='softmax')(output_layer)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model