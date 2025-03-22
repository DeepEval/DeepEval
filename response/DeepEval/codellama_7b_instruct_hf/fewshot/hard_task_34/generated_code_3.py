import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    # Define the input layer with shape (28, 28, 1)
    input_layer = Input(shape=(28, 28, 1))

    # Define the main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(main_path)

    # Define the branch path
    branch_path = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch_path)

    # Add the outputs of the main and branch paths
    adding_layer = Add()([main_path, branch_path])

    # Flatten the output
    flatten_layer = Flatten()(adding_layer)

    # Output layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model