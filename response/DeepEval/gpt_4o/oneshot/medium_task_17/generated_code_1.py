import keras
from keras.layers import Input, Permute, Reshape, Dense, Flatten

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 channels
    input_layer = Input(shape=input_shape)

    # Step 1: Reshape input tensor into shape (height, width, groups, channels_per_group)
    groups = 3
    channels_per_group = input_shape[2] // groups
    reshaped = Reshape((input_shape[0], input_shape[1], groups, channels_per_group))(input_layer)

    # Step 2: Swap the third and fourth dimensions (channel shuffling)
    shuffled = Permute((1, 2, 4, 3))(reshaped)

    # Step 3: Reshape back to the original input shape
    reshaped_back = Reshape(input_shape)(shuffled)

    # Flatten the output before passing through dense layers
    flatten_layer = Flatten()(reshaped_back)

    # Fully connected layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model