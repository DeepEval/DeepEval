import keras
from keras.layers import Input, Reshape, Permute, Lambda, Dense

def dl_model():
    # Input layer with shape (32, 32, 3) for CIFAR-10 dataset
    input_layer = Input(shape=(32, 32, 3))

    # Reshape the input tensor into (height, width, groups, channels_per_group)
    groups = 3
    channels_per_group = 3 // groups
    reshape = Reshape((32, 32, groups, channels_per_group))(input_layer)

    # Swap the third and fourth dimensions to enable channel shuffling
    permutation = Permute((1, 2, 4, 3))(reshape)

    # Reshape the tensor back to its original input shape
    original_shape = keras.backend.int_shape(reshape)
    reshape_back = Reshape(original_shape[1:])(permutation)

    # Apply a fully connected layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(reshape_back)

    # Create the Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model