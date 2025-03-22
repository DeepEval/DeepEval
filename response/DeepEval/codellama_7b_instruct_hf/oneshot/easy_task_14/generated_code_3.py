import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Global average pooling
    global_average_pooling = GlobalAveragePooling2D()(input_layer)

    # Fully connected layers
    fully_connected_1 = Dense(128, activation='relu')(global_average_pooling)
    fully_connected_2 = Dense(64, activation='relu')(fully_connected_1)

    # Reshape the weights
    reshaped_weights = Reshape((3, 3, 3))(fully_connected_2)

    # Multiply the weights with the input feature map
    weighted_input = input_layer * reshaped_weights

    # Flatten the result
    flattened_output = Flatten()(weighted_input)

    # Fully connected layer
    fully_connected_3 = Dense(10, activation='softmax')(flattened_output)

    # Create the model
    model = Model(inputs=input_layer, outputs=fully_connected_3)

    return model