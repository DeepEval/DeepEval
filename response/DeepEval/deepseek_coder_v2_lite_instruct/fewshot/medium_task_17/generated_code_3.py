import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Reshape the input tensor into three groups
    reshaped = Reshape(target_shape=(32, 32, 3, 1))(input_layer)

    # Swap the third and fourth dimensions to enable channel shuffling
    permuted = Permute((1, 2, 4, 3))(reshaped)

    # Reshape back to the original input shape
    final_reshaped = Reshape(target_shape=(32, 32, 3))(permuted)

    # Flatten the final reshaped tensor
    flattened = Flatten()(final_reshaped)

    # Pass through a fully connected layer with a softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model