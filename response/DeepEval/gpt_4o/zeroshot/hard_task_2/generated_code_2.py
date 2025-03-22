import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # CIFAR-10 images are 32x32 with 3 channels
    input_shape = (32, 32, 3)
    num_classes = 10

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Function to create a convolutional path
    def conv_path(x):
        x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
        return x

    # Apply the convolutional path to each split
    path_1 = conv_path(split_layer[0])
    path_2 = conv_path(split_layer[1])
    path_3 = conv_path(split_layer[2])

    # Combine the outputs of the three paths using addition
    main_path = Add()([path_1, path_2, path_3])

    # Fuse with the original input through another addition
    fused_output = Add()([main_path, inputs])

    # Flatten and add a fully connected layer for classification
    flat_output = Flatten()(fused_output)
    predictions = Dense(num_classes, activation='softmax')(flat_output)

    # Create the model
    model = Model(inputs=inputs, outputs=predictions)

    return model