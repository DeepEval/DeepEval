import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Branch path
    branch_path = input_layer

    # Add the outputs of both paths
    added_output = Add()([main_path, branch_path])

    # Second block
    # Max pooling layers
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(added_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(added_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(added_output)

    # Flatten the results of the pooling layers
    flattened1 = Flatten()(pool1)
    flattened2 = Flatten()(pool2)
    flattened3 = Flatten()(pool3)

    # Concatenate the flattened outputs
    concatenated_output = keras.layers.Concatenate()([flattened1, flattened2, flattened3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated_output)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model