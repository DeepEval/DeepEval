import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Reshape

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block
    x = input_layer
    x = MaxPooling2D((1, 1), strides=(1, 1))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = MaxPooling2D((4, 4), strides=(4, 4))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Second block
    x = Reshape((1, 1, 64))(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((4, 4), strides=(4, 4))(x)
    x = Dropout(0.2)(x)

    # Concatenate the outputs from all paths
    x = Concatenate()([x, x, x, x])

    # Final classification layer
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model with the Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model