from keras.models import Model
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the main path
    main_path = SeparableConv2D(32, (3, 3), activation='relu')(input_shape)
    main_path = SeparableConv2D(32, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)

    # Define the branch path
    branch_path = Conv2D(16, (1, 1), activation='relu')(input_shape)

    # Define the output from both paths
    outputs = Add()([main_path, branch_path])

    # Define the flattening layer
    flatten = Flatten()(outputs)

    # Define the fully connected layer
    fc = Dense(10, activation='softmax')(flatten)

    # Create the model
    model = Model(inputs=input_shape, outputs=fc)

    return model