from keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape, Multiply
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    x = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = GlobalAveragePooling2D()(x)

    # Define the second block
    x = Reshape((x.shape[1], x.shape[2], 1))(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Multiply()([x, x])
    x = GlobalAveragePooling2D()(x)

    # Define the output layer
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=x, outputs=x)

    return model