from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = Conv2D(32, (3, 3), activation='relu')(input_shape)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Flatten()(main_path)

    # Define the feature extraction paths
    feature_extraction_1 = Conv2D(16, (3, 3), activation='relu')(input_shape)
    feature_extraction_1 = MaxPooling2D((2, 2))(feature_extraction_1)
    feature_extraction_1 = Flatten()(feature_extraction_1)

    feature_extraction_2 = Conv2D(8, (3, 3), activation='relu')(input_shape)
    feature_extraction_2 = MaxPooling2D((2, 2))(feature_extraction_2)
    feature_extraction_2 = Flatten()(feature_extraction_2)

    # Define the branch path
    branch_path = Concatenate()([feature_extraction_1, feature_extraction_2])
    branch_path = Dense(64, activation='relu')(branch_path)
    branch_path = Dense(32, activation='relu')(branch_path)

    # Define the output path
    output_path = Concatenate()([main_path, branch_path])
    output_path = Dense(10, activation='softmax')(output_path)

    # Create the model
    model = Model(inputs=input_shape, outputs=output_path)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model