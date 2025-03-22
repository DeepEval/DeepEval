import keras
from keras.layers import Input, Lambda, Concatenate, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the main path
    main_path = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    main_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Flatten()(main_path)

    # Define the branch path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the main and branch paths
    combined_path = Concatenate()([main_path, branch_path])

    # Add a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(combined_path)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model