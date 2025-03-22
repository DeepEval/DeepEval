import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input shape for the model
    input_shape = (32, 32, 3)

    # Define the main path of the model
    main_path = Input(shape=input_shape)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = Concatenate()(main_path, main_path[:, :, :, 1:])
    main_path = Conv2D(128, (3, 3), activation='relu')(main_path)
    main_path = Concatenate()(main_path, main_path[:, :, :, 1:])
    main_path = Conv2D(256, (3, 3), activation='relu')(main_path)
    main_path = Concatenate()(main_path, main_path[:, :, :, 1:])
    main_path = Conv2D(512, (3, 3), activation='relu')(main_path)
    main_path = Flatten()(main_path)

    # Define the branch path of the model
    branch_path = Input(shape=input_shape)
    branch_path = Conv2D(32, (1, 1), activation='relu')(branch_path)
    branch_path = MaxPooling2D((2, 2))(branch_path)
    branch_path = Conv2D(64, (3, 3), activation='relu')(branch_path)
    branch_path = Flatten()(branch_path)

    # Define the model
    model = keras.models.Model(inputs=[main_path, branch_path], outputs=Concatenate()([main_path, branch_path]))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model