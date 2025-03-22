from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D


def dl_model():
    
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first block
    block1 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    block1 = Conv2D(32, (3, 3), activation='relu')(block1)
    block1 = MaxPooling2D((2, 2))(block1)
    block1 = Flatten()(block1)
    block1 = Dropout(0.25)(block1)

    # Define the second block
    block2 = Conv2D(64, (3, 3), activation='relu')(input_shape)
    block2 = Conv2D(64, (3, 3), activation='relu')(block2)
    block2 = MaxPooling2D((2, 2))(block2)
    block2 = Flatten()(block2)
    block2 = Dropout(0.25)(block2)

    # Define the fully connected layers
    fc1 = Dense(128, activation='relu')(block1)
    fc2 = Dense(128, activation='relu')(fc1)
    fc3 = Dense(10, activation='softmax')(fc2)

    # Define the model
    model = Model(inputs=input_shape, outputs=fc3)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the model
    return model