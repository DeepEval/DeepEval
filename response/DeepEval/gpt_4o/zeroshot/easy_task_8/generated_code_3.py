from tensorflow.keras.layers import Input, SeparableConv2D, Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input shape for MNIST dataset (28x28 pixels, grayscale)
    input_shape = (28, 28, 1)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Depthwise separable convolutional layer with dropout
    x = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Dropout(0.25)(x)

    # 1x1 convolutional layer with dropout
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Dropout(0.25)(x)

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to use the model
# model = dl_model()
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])