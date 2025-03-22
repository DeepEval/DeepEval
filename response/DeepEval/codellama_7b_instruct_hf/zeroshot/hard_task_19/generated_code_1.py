import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Add


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    x = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Define the branch path
    y = GlobalAveragePooling2D()(x)
    y = Dense(64, activation='relu')(y)
    y = Dense(64, activation='relu')(y)
    y = Flatten()(y)

    # Add the outputs from both paths
    z = Add()([x, y])

    # Apply additional fully connected layers
    z = Dense(128, activation='relu')(z)
    z = Dense(10, activation='softmax')(z)

    # Create the model
    model = Model(inputs=x, outputs=z)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model