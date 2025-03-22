import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Reshape input tensor into three groups
    reshaped_input = Reshape((32, 32, 3, 1))(input_layer)

    # Swap the third and fourth dimensions using a permutation operation
    permuted_input = Permute((0, 1, 3, 2))(reshaped_input)

    # Reshape back to the original input shape
    flattened_input = Flatten()(permuted_input)

    # Fully connected layer with softmax activation for classification
    output_layer = Dense(10, activation='softmax')(flattened_input)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model