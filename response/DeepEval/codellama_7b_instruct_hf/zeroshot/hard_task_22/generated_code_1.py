import keras
from keras.models import Model
from keras.layers import Input, Lambda, Split, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = []

    # Split the input into three groups along the channel
    main_path.append(Lambda(lambda x: tf.split(x, 3, axis=3))(input_shape))

    # Apply multi-scale feature extraction with separable convolutional layers of varying kernel sizes
    for i in range(3):
        main_path.append(Conv2D(64, (1, 1), use_bias=False, kernel_initializer='he_normal')(main_path[i]))
        main_path.append(BatchNormalization()(main_path[i]))
        main_path.append(Activation('relu')(main_path[i]))
        main_path.append(Conv2D(64, (3, 3), use_bias=False, kernel_initializer='he_normal')(main_path[i]))
        main_path.append(BatchNormalization()(main_path[i]))
        main_path.append(Activation('relu')(main_path[i]))
        main_path.append(Conv2D(64, (5, 5), use_bias=False, kernel_initializer='he_normal')(main_path[i]))
        main_path.append(BatchNormalization()(main_path[i]))
        main_path.append(Activation('relu')(main_path[i]))

    # Concatenate the outputs from the three groups
    main_path.append(Concatenate(axis=3)(main_path[3:]))

    # Apply a 1x1 convolutional layer to align the number of output channels with those of the branch path
    main_path.append(Conv2D(64, (1, 1), use_bias=False, kernel_initializer='he_normal')(main_path[-1]))
    main_path.append(BatchNormalization()(main_path[-1]))
    main_path.append(Activation('relu')(main_path[-1]))

    # Define the branch path
    branch_path = []

    # Apply a 1x1 convolutional layer to the input
    branch_path.append(Conv2D(64, (1, 1), use_bias=False, kernel_initializer='he_normal')(input_shape))
    branch_path.append(BatchNormalization()(branch_path[-1]))
    branch_path.append(Activation('relu')(branch_path[-1]))

    # Apply a 3x3 convolutional layer
    branch_path.append(Conv2D(64, (3, 3), use_bias=False, kernel_initializer='he_normal')(branch_path[-1]))
    branch_path.append(BatchNormalization()(branch_path[-1]))
    branch_path.append(Activation('relu')(branch_path[-1]))

    # Fuse the outputs from both paths through addition
    output = Add()([main_path[-1], branch_path[-1]])

    # Flatten the output and pass through two fully connected layers for classification
    output = Flatten()(output)
    output = Dense(64, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)

    # Create the model
    model = Model(inputs=input_shape, outputs=output)

    # Return the model
    return model