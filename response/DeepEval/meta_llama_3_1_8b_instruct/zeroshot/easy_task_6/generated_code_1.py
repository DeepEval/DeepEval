# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Add


def dl_model():
    """
    Create a deep learning model with two paths for image classification using the MNIST dataset.

    Returns:
        model (Model): The constructed model.
    """

    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the inputs
    main_input = Input(shape=input_shape)
    branch_input = Input(shape=input_shape)

    # Define the main path
    main_path = Conv2D(32, (3, 3), activation='relu')(main_input)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Conv2D(1, (3, 3), activation='relu')(main_path)
    main_path = UpSampling2D((2, 2))(main_path)

    # Define the branch path
    branch_path = main_input

    # Combine the two paths
    combined_path = Add()([main_path, branch_path])

    # Flatten the combined path
    x = Flatten()(combined_path)

    # Define the fully connected layer
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=[main_input, branch_input], outputs=x)

    return model


# Create the model
model = dl_model()
model.summary()