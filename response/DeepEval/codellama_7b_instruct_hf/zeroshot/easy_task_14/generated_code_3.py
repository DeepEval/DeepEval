from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first fully connected layer with 128 units
    x = Input(shape=input_shape)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)

    # Define the second fully connected layer with 10 units
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=x, outputs=x)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model