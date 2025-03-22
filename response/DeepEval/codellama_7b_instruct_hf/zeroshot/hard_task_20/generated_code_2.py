from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Add
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = Lambda(lambda x: tf.split(x, 3, axis=3))(input_shape)
    main_path = Conv2D(32, (1, 1), activation='relu')(main_path)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = Conv2D(32, (5, 5), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Flatten()(main_path)
    main_path = Dense(128, activation='relu')(main_path)
    main_path = Dense(10, activation='softmax')(main_path)

    # Define the branch path
    branch_path = Lambda(lambda x: tf.split(x, 3, axis=3))(input_shape)
    branch_path = Conv2D(32, (1, 1), activation='relu')(branch_path)
    branch_path = MaxPooling2D((2, 2))(branch_path)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(128, activation='relu')(branch_path)
    branch_path = Dense(10, activation='softmax')(branch_path)

    # Combine the main and branch paths through addition
    outputs = Add()([main_path, branch_path])

    # Define the model
    model = Model(inputs=input_shape, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model