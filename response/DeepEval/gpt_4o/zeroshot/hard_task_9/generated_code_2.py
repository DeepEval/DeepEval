from tensorflow.keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 channels

    # Input layer
    inputs = Input(shape=input_shape)

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)

    # Branch 2: 1x1 convolution followed by a 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)

    # Branch 3: 1x1 convolution followed by two 3x3 convolutions
    branch3 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)

    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolution to adjust output dimensions to match input channel size
    adjusted_output = Conv2D(3, (1, 1), activation='relu', padding='same')(concatenated)

    # Adding the input directly to the adjusted output
    fused = Add()([inputs, adjusted_output])

    # Flatten the output for the fully connected layers
    flattened = Flatten()(fused)

    # Fully connected layers
    dense1 = Dense(256, activation='relu')(flattened)
    dense2 = Dense(128, activation='relu')(dense1)
    dense3 = Dense(10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=inputs, outputs=dense3)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of usage
if __name__ == "__main__":
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Normalize data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Build and summarize the model
    model = dl_model()
    model.summary()

    # Train the model (example)
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))