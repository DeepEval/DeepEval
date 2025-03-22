from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications.cifar10 import Cifar10

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the four convolutional paths
    conv1x1 = Conv2D(32, (1, 1), activation='relu')
    conv3x3 = Conv2D(32, (3, 3), activation='relu')
    conv3x3_2 = Conv2D(32, (3, 3), activation='relu')
    conv_max_pool = MaxPooling2D((2, 2))

    # Define the four parallel convolutional paths
    conv_path_1 = conv1x1(input_shape)
    conv_path_2 = conv3x3(conv_path_1)
    conv_path_3 = conv3x3_2(conv_path_1)
    conv_path_4 = conv_max_pool(input_shape)

    # Concatenate the outputs of the four paths
    concatenated = Concatenate()([conv_path_1, conv_path_2, conv_path_3, conv_path_4])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Add a dense layer with 128 units
    dense = Dense(128, activation='relu')(flattened)

    # Add a dropout layer with a dropout rate of 0.2
    dropout = Dropout(0.2)(dense)

    # Add a final output layer with softmax activation
    output = Dense(10, activation='softmax')(dropout)

    # Create the model
    model = Model(inputs=input_shape, outputs=output)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model