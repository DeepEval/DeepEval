from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)

    # Define path 1
    path1_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(path1_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)

    # Define path 2
    path2_input = Input(shape=input_shape)
    y = Conv2D(32, (3, 3), activation='relu')(path2_input)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = GlobalAveragePooling2D()(y)

    # Define the addition layer
    z = Add()([x, y])

    # Define the fully connected layer
    z = Flatten()(z)
    z = Dense(10, activation='softmax')(z)

    # Define the model
    model = Model(inputs=[path1_input, path2_input], outputs=z)

    return model