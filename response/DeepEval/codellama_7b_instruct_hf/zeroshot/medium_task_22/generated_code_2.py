import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the multi-branch convolutional architecture
    branch1 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    branch2 = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch3 = MaxPooling2D((2, 2))(input_shape)

    # Concatenate the branches
    x = keras.layers.concatenate([branch1, branch2, branch3], axis=1)

    # Flatten the features
    x = Flatten()(x)

    # Add fully connected layers for classification
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_shape, outputs=x)

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Return the model
    return model


    model = dl_model((32, 32, 3))

    return model