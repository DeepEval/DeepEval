import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, concatenate, Dense, Flatten, Input

def dl_model():
    # Define the input layer
    img_input = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(img_input)
    x = MaxPooling2D()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)

    # Branch path
    y = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(img_input)
    y = MaxPooling2D()(y)

    # Combine outputs
    combined = concatenate([x, y])

    # Flatten and fully connected layers
    x = Flatten()(combined)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=img_input, outputs=x)

    return model