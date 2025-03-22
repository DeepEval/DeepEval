from keras.layers import Input, Dense, Conv2D, DepthwiseSeparableConv2D, Flatten, Permute, Reshape
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = DepthwiseSeparableConv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)

    # Block 2
    x = Flatten()(x)
    x = Permute((3, 1, 2))(x)
    x = Reshape((28, 28, 4, 8))(x)
    x = Permute((1, 2, 3, 4))(x)
    x = Reshape((28, 28, 8))(x)

    # Output layer
    output_layer = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model