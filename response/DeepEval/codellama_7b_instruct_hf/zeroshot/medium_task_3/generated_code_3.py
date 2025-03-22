from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first sequential block
    x = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Define the second sequential block
    y = Input(shape=input_shape)
    y = Conv2D(32, (3, 3), activation='relu')(y)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(128, (3, 3), activation='relu')(y)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(256, (3, 3), activation='relu')(y)
    y = MaxPooling2D((2, 2))(y)
    y = Flatten()(y)

    # Combine the outputs from both paths
    z = Add()([x, y])

    # Define the fully connected layer
    z = Flatten()(z)
    z = Dense(10, activation='softmax')(z)

    # Create the model
    model = Model(inputs=[x, y], outputs=z)

    return model