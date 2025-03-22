import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first sequential block
    block_1 = Sequential()
    block_1.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    block_1.add(MaxPooling2D((2, 2)))

    # Define the second sequential block
    block_2 = Sequential()
    block_2.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    block_2.add(MaxPooling2D((2, 2)))

    # Combine the outputs from both blocks
    combined_output = Concatenate()([block_1.output, block_2.output])

    # Add a batch normalization layer
    batch_norm = BatchNormalization()(combined_output)

    # Flatten the output
    flatten_output = Flatten()(batch_norm)

    # Add a fully connected layer
    dense = Dense(units=128, activation='relu')(flatten_output)

    # Add another fully connected layer
    dense_1 = Dense(units=64, activation='relu')(dense)

    # Add the final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense_1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model