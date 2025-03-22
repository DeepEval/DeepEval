import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Permute, Reshape

def dl_model():
    input_shape = (32, 32, 3)  # Assuming the CIFAR-10 images are 32x32 pixels
    num_groups = 3
    channels_per_group = 3  # Assuming RGB images

    # Input layer
    input_layer = Input(shape=input_shape)

    # Reshape to separate channels and group into three
    reshaped_input = Permute((2, 3, 1))(input_layer)
    reshaped_input = Reshape((input_shape[0] * input_shape[1] * num_groups, channels_per_group))(reshaped_input)

    # Shuffle channels
    shuffled_input = Permute((2, 3, 1))(input_layer)
    shuffled_input = Reshape((input_shape[0] * input_shape[1] * num_groups, channels_per_group))(shuffled_input)

    # Back to original shape
    input_layer = Reshape(input_shape)(reshaped_input)
    input_layer = Permute((2, 3, 1))(input_layer)

    # Add Conv2D layers
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Additional layers for the block
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Flatten and fully connected layers
    flat1 = Flatten()(pool2)
    dense1 = Dense(256, activation='relu')(flat1)
    dense2 = Dense(128, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.summary()