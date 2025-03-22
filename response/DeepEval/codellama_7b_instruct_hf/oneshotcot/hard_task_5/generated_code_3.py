import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Permute
from keras.models import Model
from keras.applications.vgg16 import VGG16


def dl_model():

    # Load the VGG-16 model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the pre-trained layers
    vgg_model.trainable = False

    # Add a new input layer for the main path
    main_input = Input(shape=(32, 32, 3))

    # Add a new input layer for the branch
    branch_input = Input(shape=(32, 32, 3))

    # Define the main path
    main_path = Conv2D(64, (3, 3), activation='relu')(main_input)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)

    # Split the main path into three groups
    main_path = Lambda(lambda x: tf.split(x, 3, axis=3))(main_path)

    # Process each group separately
    main_path_1 = Conv2D(64, (1, 1), activation='relu')(main_path[0])
    main_path_2 = Conv2D(64, (3, 3), activation='relu')(main_path[1])
    main_path_3 = Conv2D(64, (5, 5), activation='relu')(main_path[2])

    # Concatenate the outputs from each group
    main_path = Concatenate()([main_path_1, main_path_2, main_path_3])

    # Add batch normalization and flatten the result
    main_path = BatchNormalization()(main_path)
    main_path = Flatten()(main_path)

    # Add a new dense layer with 64 units
    main_path = Dense(64, activation='relu')(main_path)

    # Add the branch to the main path
    branch_path = Conv2D(64, (3, 3), activation='relu')(branch_input)
    branch_path = MaxPooling2D((2, 2))(branch_path)
    branch_path = Conv2D(64, (3, 3), activation='relu')(branch_path)
    branch_path = MaxPooling2D((2, 2))(branch_path)
    branch_path = Conv2D(64, (3, 3), activation='relu')(branch_path)

    # Add batch normalization and flatten the result
    branch_path = BatchNormalization()(branch_path)
    branch_path = Flatten()(branch_path)

    # Add a new dense layer with 64 units
    branch_path = Dense(64, activation='relu')(branch_path)

    # Add the main path and branch to the input
    output = Concatenate()([main_path, branch_path])

    # Add a new dense layer with 10 units
    output = Dense(10, activation='softmax')(output)

    # Create the model
    model = Model(inputs=[main_input, branch_input], outputs=output)


    return model