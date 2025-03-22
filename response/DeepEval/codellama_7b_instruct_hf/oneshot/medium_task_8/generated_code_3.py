from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16


def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the input data
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Define the main path of the model
    main_path = Input(shape=(32, 32, 3))

    # Split the input into three groups
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(main_path)

    # Process the first group
    unchanged_group = split_layer[0]

    # Process the second group
    conv_layer = Conv2D(64, (3, 3), padding='same', activation='relu')(split_layer[1])
    max_pooling_layer = MaxPooling2D((2, 2), padding='same')(conv_layer)

    # Process the third group
    conv_layer_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(split_layer[2])
    max_pooling_layer_2 = MaxPooling2D((2, 2), padding='same')(conv_layer_2)

    # Combine the output of the second and third groups
    combined_output = Concatenate()([max_pooling_layer, max_pooling_layer_2])

    # Apply batch normalization and flatten the output
    batch_normalized_output = BatchNormalization()(combined_output)
    flattened_output = Flatten()(batch_normalized_output)

    # Define the branch path of the model
    branch_path = Input(shape=(32, 32, 3))

    # Process the input using a 1x1 convolutional layer
    conv_layer_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(branch_path)

    # Fuse the output of the main and branch paths
    fused_output = Add()([combined_output, conv_layer_3])

    # Define the final classification layer
    final_layer = Dense(10, activation='softmax')(fused_output)

    # Create the model
    model = Model(inputs=[main_path, branch_path], outputs=final_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    return model