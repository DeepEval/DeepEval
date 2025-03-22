import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape, Multiply


def dl_model():
    
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    x = Input(shape=input_shape)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Reshape(target_shape=input_shape)(x)
    x = Multiply()([x, Input(shape=input_shape)])

    # Define the second block
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Define the fused output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=x, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    print(model.summary())

    # Return the model
    return model