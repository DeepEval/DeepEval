from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense



def dl_model():
    # Define the main pathway
    input_layer = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    # Define the branch pathway
    branch_input = Input(shape=(28, 28, 1))
    branch_x = Conv2D(32, (3, 3), activation='relu')(branch_input)
    branch_x = Conv2D(64, (1, 1), activation='relu')(branch_x)
    branch_x = MaxPooling2D((2, 2))(branch_x)

    # Add the two pathways and fuse the outputs
    main_output = x
    branch_output = branch_x
    output = Add()([main_output, branch_output])

    # Flatten and fc layers
    flatten = Flatten()(output)
    fc = Dense(128, activation='relu')(flatten)
    fc = Dropout(0.5)(fc)

    # Output layer
    output_layer = Dense(10, activation='softmax')(fc)

    # Define the model
    model = Model(inputs=[input_layer, branch_input], outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model