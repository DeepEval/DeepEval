import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the feature extraction branches
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch1 = Conv2D(64, (1, 1), activation='relu')(branch1)
    branch2 = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch3 = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)

    # Concatenate the output of the feature extraction branches
    x = keras.layers.concatenate([branch1, branch2, branch3], axis=1)

    # Adjust the output dimensions to match the input image's channel size
    x = Conv2D(3, (1, 1), activation='relu')(x)

    # Add the branch directly connected to the input
    x = keras.layers.add([x, input_shape])

    # Define the fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_shape, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model