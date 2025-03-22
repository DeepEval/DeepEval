import numpy as np
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense, Concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical

def build_branch(input_tensor):
    # Depthwise separable convolution layer
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
    # 1x1 convolution layer
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
    # Dropout layer
    x = Dropout(0.5)(x)
    return x

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Three branches
    branch1 = build_branch(input_layer)
    branch2 = build_branch(input_layer)
    branch3 = build_branch(input_layer)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    x = Flatten()(concatenated)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout for regularization
    x = Dense(10, activation='softmax')(x)  # Final classification layer

    # Create the model
    model = Model(inputs=input_layer, outputs=x)

    return model

# Example of how to use the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Load and prepare the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0  # Normalize and reshape
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))