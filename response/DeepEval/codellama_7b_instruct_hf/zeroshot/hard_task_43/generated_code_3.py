from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Concatenate
from keras.models import Model
from keras.applications import MobileNetV2

def dl_model():
    # Define input shape
    input_shape = (28, 28, 1)

    # Define the first block
    first_block = []
    for i in range(3):
        if i == 0:
            first_block.append(
                Conv2D(
                    64,
                    kernel_size=(1, 1),
                    strides=(2, 2),
                    activation='relu'
                )
            )
        elif i == 1:
            first_block.append(
                Conv2D(
                    64,
                    kernel_size=(2, 2),
                    strides=(2, 2),
                    activation='relu'
                )
            )
        elif i == 2:
            first_block.append(
                Conv2D(
                    64,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    activation='relu'
                )
            )
    first_block = Concatenate(axis=1)(first_block)
    first_block = Flatten()(first_block)
    first_block = Dense(64, activation='relu')(first_block)
    first_block = Reshape((4, 4, 64))(first_block)

    # Define the second block
    second_block = []
    for i in range(3):
        if i == 0:
            second_block.append(
                Conv2D(
                    64,
                    kernel_size=(1, 1),
                    strides=(2, 2),
                    activation='relu'
                )
            )
        elif i == 1:
            second_block.append(
                Conv2D(
                    64,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    activation='relu'
                )
            )
        elif i == 2:
            second_block.append(
                Conv2D(
                    64,
                    kernel_size=(7, 7),
                    strides=(2, 2),
                    activation='relu'
                )
            )
    second_block = Concatenate(axis=1)(second_block)
    second_block = Flatten()(second_block)
    second_block = Dense(64, activation='relu')(second_block)
    second_block = Reshape((4, 4, 64))(second_block)

    # Define the output layer
    output_layer = Dense(10, activation='softmax')(second_block)

    # Define the model
    model = Model(inputs=first_block, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model