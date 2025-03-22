import keras
from keras.layers import Input, Dense, Reshape, Permute, Flatten
from keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels

    input_layer = Input(shape=input_shape)

    # Reshape to (height, width, groups, channels_per_group)
    groups = 3
    channels_per_group = input_shape[2] // groups
    reshaped = Reshape((input_shape[0], input_shape[1], groups, channels_per_group))(input_layer)

    # Permute dimensions to enable channel shuffling
    shuffled = Permute((1, 2, 4, 3))(reshaped)

    # Reshape back to original input shape
    reshaped_back = Reshape(input_shape)(shuffled)

    # Flatten and pass through fully connected layer for classification
    flatten_layer = Flatten()(reshaped_back)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Usage
model = dl_model()
model.summary()