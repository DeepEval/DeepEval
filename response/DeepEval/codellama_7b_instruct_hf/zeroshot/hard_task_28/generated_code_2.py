from keras.applications import VGG16
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model


def dl_model():

    # Main path
    main_path = VGG16.get_layer('block5_conv3').output
    main_path = Conv2D(256, (7, 7), activation='relu', padding='same')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Flatten()(main_path)

    # Branch path
    branch_path = VGG16.get_layer('block1_conv1').output
    branch_path = Conv2D(256, (1, 1), activation='relu')(branch_path)
    branch_path = Conv2D(256, (1, 1), activation='relu')(branch_path)
    branch_path = Flatten()(branch_path)

    # Combine outputs
    output = main_path + branch_path

    # Fully connected layers
    output = Dense(128, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)

    # Create model
    model = Model(inputs=main_path, outputs=output)

    # Compile model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model