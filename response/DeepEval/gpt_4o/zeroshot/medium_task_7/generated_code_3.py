from tensorflow.keras.layers import Input, Conv2D, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First path: three sequential convolutional layers
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x3 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)

    # Second path: a separate convolutional layer processing the input directly
    y = Conv2D(128, (3, 3), activation='relu', padding='same')(input_layer)

    # Adding the outputs of the paths
    added = Add()([x1, x2, x3, y])

    # Flatten and pass through fully connected layers
    flattened = Flatten()(added)
    fc1 = Dense(128, activation='relu')(flattened)
    output_layer = Dense(10, activation='softmax')(fc1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Load CIFAR-10 dataset for a quick test
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Example: Fitting the model (uncomment for actual training)
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))