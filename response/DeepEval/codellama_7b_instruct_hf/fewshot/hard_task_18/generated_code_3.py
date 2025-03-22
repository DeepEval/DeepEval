import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense

 和 return model
def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    block1 = keras.Sequential()
    block1.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    block1.add(Conv2D(32, (3, 3), activation='relu'))
    block1.add(AveragePooling2D((2, 2)))

    # Define the second block
    block2 = keras.Sequential()
    block2.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    block2.add(Conv2D(32, (3, 3), activation='relu'))
    block2.add(AveragePooling2D((2, 2)))

    # Define the main path
    main_path = Add()([block1, block2])

    # Define the branch path
    branch_path = Conv2D(32, (3, 3), activation='relu')(main_path)

    # Define the output layer
    output_layer = Flatten()(branch_path)
    output_layer = Dense(10, activation='softmax')(output_layer)

    # Create the model
    model = keras.Model(inputs=input_shape, outputs=output_layer)

    return model