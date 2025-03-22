from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Add


def dl_model():
    
    # Block 1
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    branch1 = x

    x = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = x

    # Block 2
    x = MaxPooling2D((2, 2))(branch1)
    x = MaxPooling2D((4, 4))(x)
    x = MaxPooling2D((8, 8))(x)
    flattened = Flatten()(x)

    # Block 3
    x = Add()([branch2, flattened])
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=x)

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Return the constructed model
    return model