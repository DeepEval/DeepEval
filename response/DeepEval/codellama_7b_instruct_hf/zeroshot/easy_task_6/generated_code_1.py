import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, Add

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first path
    x = Input(shape=input_shape)
    y = Conv2D(32, (3, 3), activation='relu')(x)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = Add()([x, y])

    # Define the branch path
    z = Input(shape=input_shape)
    w = Conv2D(32, (3, 3), activation='relu')(z)
    w = Conv2D(64, (3, 3), activation='relu')(w)
    w = Add()([z, w])

    # Combine the two paths through an addition operation
    y = Add()([y, w])

    # Flatten the output and add a fully connected layer
    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)
    y = Dense(10, activation='softmax')(y)

    # Define the model
    model = Model(inputs=[x, z], outputs=y)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model