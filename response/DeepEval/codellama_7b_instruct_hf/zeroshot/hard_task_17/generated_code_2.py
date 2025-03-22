from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten, Conv2D, MaxPooling2D, Add

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first sequential block
    block1 = Sequential([
        GlobalAveragePooling2D(input_shape=input_shape),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Define the second sequential block
    block2 = Sequential([
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten()
    ])

    # Define the main path of the model
    main_path = block1(block2.output)

    # Define the branch of the model
    branch = block1(main_path)

    # Define the fusion layer
    fusion = Add()([main_path, branch])

    # Define the output layer
    output = Dense(10, activation='softmax')(fusion)

    # Create the model
    model = Model(inputs=block2.input, outputs=output)

    return model