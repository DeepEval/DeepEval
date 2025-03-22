import keras
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Concatenate
from keras.models import Model

，增加dl_model 和 return model
def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the main path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(128, activation='relu')(main_path)
    main_path = Flatten()(main_path)
    main_path = Dense(32, activation='relu')(main_path)

    # Define the branch path
    branch_path = input_layer

    # Add the outputs from both paths
    output = Concatenate()([main_path, branch_path])

    # Apply two fully connected layers to produce the final probability distribution
    output = Dense(10, activation='softmax')(output)

    # Define the model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model