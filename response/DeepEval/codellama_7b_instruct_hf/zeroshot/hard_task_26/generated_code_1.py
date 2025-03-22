import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from keras.models import Model


def dl_model():

    # Define the main path
    main_input = Input(shape=(32, 32, 3))
    x = Conv2D(32, (1, 1), activation='relu')(main_input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)

    # Define the branch path
    branch_input = Input(shape=(32, 32, 3))
    y = Conv2D(32, (1, 1), activation='relu')(branch_input)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = UpSampling2D((2, 2))(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = UpSampling2D((2, 2))(y)
    y = Conv2D(32, (3, 3), activation='relu')(y)
    y = UpSampling2D((2, 2))(y)
    y = Conv2D(32, (3, 3), activation='relu')(y)

    # Merge the outputs of the main and branch paths
    z = Concatenate()([x, y])
    z = Flatten()(z)
    z = Dense(64, activation='relu')(z)
    z = Dense(10, activation='softmax')(z)

    # Create the model
    model = Model(inputs=[main_input, branch_input], outputs=z)

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model