import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.cifar10 import Cifar10

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = Cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = Input(shape=input_shape)
    main_path_output = main_path

    # Split the input into three groups along the last dimension
    split_main_path_output = Lambda(lambda x: tf.split(x, 3, axis=-1))(main_path_output)

    # Extract features from the second group using a 3x3 convolutional layer
    second_group = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(split_main_path_output[1])

    # Combine the second group with the third group
    combined_group = Concatenate()([split_main_path_output[1], split_main_path_output[2]])

    # Pass the combined group through an additional 3x3 convolutional layer
    combined_group = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(combined_group)

    # Concatenate the outputs of all three groups
    main_path_output = Concatenate()(split_main_path_output)

    # Define the branch path
    branch_path = Input(shape=input_shape)
    branch_path_output = branch_path

    # Apply a 1x1 convolutional layer to the input
    branch_path_output = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(branch_path_output)

    # Fuse the outputs from both the main and branch paths through addition
    fused_output = Lambda(lambda x: tf.add(x[0], x[1]))([main_path_output, branch_path_output])

    # Flatten the fused output and pass it through a fully connected layer
    flattened_output = Flatten()(fused_output)
    output = Dense(10, activation='softmax')(flattened_output)

    # Define the model
    model = keras.Model(inputs=[main_path, branch_path], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model