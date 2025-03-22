from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, DepthwiseSeparableConv2D, Conv2DTranspose, Permute, Reshape, Concatenate
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the primary path
    primary_path = Conv2D(32, (3, 3), activation='relu', padding='same')(input_shape)
    primary_path = DepthwiseSeparableConv2D(32, (3, 3), activation='relu', padding='same')(primary_path)
    primary_path = Conv2D(32, (3, 3), activation='relu', padding='same')(primary_path)

    # Define the branch path
    branch_path = Conv2D(32, (3, 3), activation='relu', padding='same')(input_shape)
    branch_path = DepthwiseSeparableConv2D(32, (3, 3), activation='relu', padding='same')(branch_path)
    branch_path = Conv2D(32, (3, 3), activation='relu', padding='same')(branch_path)

    # Concatenate the primary and branch paths
    concatenated_path = Concatenate()([primary_path, branch_path])

    # Reshape the concatenated path
    reshaped_path = Reshape((-1, 64))(concatenated_path)

    # Add a fully connected layer
    fc_layer = Dense(128, activation='relu')(reshaped_path)
    fc_layer = Dropout(0.2)(fc_layer)
    fc_layer = Dense(10, activation='softmax')(fc_layer)

    # Define the model
    model = Model(inputs=input_shape, outputs=fc_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model