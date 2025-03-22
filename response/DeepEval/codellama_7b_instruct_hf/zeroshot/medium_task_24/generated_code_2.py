import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first branch
    branch1 = Input(shape=input_shape)
    x = Conv2D(32, (1, 1), activation='relu')(branch1)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)

    # Define the second branch
    branch2 = Input(shape=input_shape)
    x = Conv2D(32, (1, 1), activation='relu')(branch2)
    x = Conv2D(32, (1, 7), activation='relu')(x)
    x = Conv2D(32, (7, 1), activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)

    # Define the third branch
    branch3 = Input(shape=input_shape)
    x = MaxPooling2D((2, 2))(branch3)
    x = Dropout(0.2)(x)

    # Concatenate the outputs from all branches
    x = keras.layers.concatenate([x, x, x], axis=1)

    # Flatten the output
    x = Flatten()(x)

    # Add fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=[branch1, branch2, branch3], outputs=x)

    return model