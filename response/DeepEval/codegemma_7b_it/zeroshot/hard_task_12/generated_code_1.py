import keras
from keras.models import Model
from keras.layers import Input, Conv2D, concatenate, add, Flatten, Dense

def dl_model():

    # Define the input shape
    input_shape = (32, 32, 64)

    # Create the input layer
    input_img = Input(shape=input_shape)

    # Main path
    x = Conv2D(64, (1, 1), padding='same')(input_img)
    x1 = Conv2D(16, (1, 1), padding='same')(x)
    x2 = Conv2D(16, (3, 3), padding='same')(x)
    x = concatenate([x1, x2])

    # Branch path
    y = Conv2D(16, (3, 3), padding='same')(input_img)

    # Combine the outputs
    z = add([x, y])

    # Classification head
    z = Flatten()(z)
    z = Dense(64, activation='relu')(z)
    output = Dense(10, activation='softmax')(z)

    # Create the model
    model = Model(inputs=input_img, outputs=output)

    return model