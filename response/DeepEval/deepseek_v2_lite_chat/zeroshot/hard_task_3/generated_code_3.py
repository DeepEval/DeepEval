import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Lambda, Dropout


def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Split the input into three groups along the channel dimension
    input_shape = x_train.shape[1:]
    num_channels = input_shape[2]
    split_index = num_channels // 3

    input_tensor = Input(shape=input_shape)

    # Split the input into three groups
    x = Lambda(lambda tensors: tf.split(tensors, split_index, axis=-1))(input_tensor)

    # Process the first group through a 1x1 convolution
    x1 = Conv2D(filters=64, kernel_size=1, activation='relu')(x[0])
    # Process the second group through a 3x3 convolution
    x2 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x[1])
    # Process the third group through a 3x3 convolution
    x3 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x[2])

    # Apply dropout for feature selection
    x1 = Dropout(0.2)(x1)
    x2 = Dropout(0.2)(x2)
    x3 = Dropout(0.2)(x3)

    # Concatenate the outputs from the three groups
    x = Concatenate(axis=-1)([x1, x2, x3])

    # Process the input through a 1x1 convolution to match the output dimension of the main pathway
    x_branch = Conv2D(filters=64, kernel_size=1, activation='relu')(input_tensor)

    # Process the input through another 1x1 convolution to match the output dimension of the main pathway
    x_branch = Conv2D(filters=64, kernel_size=1, activation='relu')(x_branch)

    # Combine the outputs from both pathways
    output = Concatenate()([x, x_branch])

    # Pass through a fully connected layer
    output = Flatten()(output)
    output = Dense(units=10, activation='softmax')(output)

    # Create the model
    model = Model(inputs=input_tensor, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Display the model summary
    model.summary()

    # Return the model
    return model

    # Instantiate and return the model
    model = dl_model()