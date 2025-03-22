from keras.applications import VGG16
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Add, Flatten, Dense
from keras.models import Model


def dl_model():

    input_shape = (32, 32, 3)
    num_classes = 10

    # Define the main path
    main_path = VGG16(input_shape, include_top=False, weights='imagenet')
    main_path.trainable = False

    # Define the branch path
    branch_path = Input(shape=input_shape)
    branch_path = Conv2D(32, (3, 3), activation='relu', padding='same')(branch_path)
    branch_path = MaxPooling2D((2, 2), padding='same')(branch_path)
    branch_path = Conv2D(64, (3, 3), activation='relu', padding='same')(branch_path)
    branch_path = MaxPooling2D((2, 2), padding='same')(branch_path)
    branch_path = Conv2D(128, (3, 3), activation='relu', padding='same')(branch_path)
    branch_path = MaxPooling2D((2, 2), padding='same')(branch_path)
    branch_path = Conv2D(256, (3, 3), activation='relu', padding='same')(branch_path)
    branch_path = MaxPooling2D((2, 2), padding='same')(branch_path)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(1024, activation='relu')(branch_path)
    branch_path = Dense(num_classes, activation='softmax')(branch_path)

    # Define the concatenate layer
    concat_layer = Concatenate(axis=1)([main_path.output, branch_path])

    # Define the final output
    output_layer = Conv2D(1, (1, 1), activation='relu')(concat_layer)
    output_layer = Flatten()(output_layer)
    output_layer = Dense(num_classes, activation='softmax')(output_layer)

    # Create the model
    model = Model(inputs=main_path.input, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model