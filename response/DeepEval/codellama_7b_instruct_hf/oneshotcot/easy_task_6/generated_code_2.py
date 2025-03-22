import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(main_path)

    # Define the branch path
    branch_path = Input(shape=(28, 28, 1))

    # Combine the two paths
    combined_path = Concatenate()([main_path, branch_path])

    # Add a flatten layer
    flatten_layer = Flatten()(combined_path)

    # Add a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Return the constructed model
    return model