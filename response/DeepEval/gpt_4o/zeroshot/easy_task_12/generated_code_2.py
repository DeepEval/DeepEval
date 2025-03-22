from tensorflow.keras.layers import Input, SeparableConv2D, MaxPooling2D, Conv2D
from tensorflow.keras.layers import ReLU, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def dl_model(input_shape=(28, 28, 1), num_classes=10):
    # Define the input layer
    inputs = Input(shape=input_shape)

    # Main path
    x = SeparableConv2D(32, (3, 3), padding='same')(inputs)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    # Branch path
    branch = Conv2D(64, (1, 1), padding='same')(inputs)

    # Combine main and branch paths
    x = Add()([x, branch])

    # Classification head
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., None]  # Add channel dimension
x_test = x_test[..., None]    # Add channel dimension

# Convert labels to one-hot encoded format
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Instantiate the model
model = dl_model()

# Print the model summary
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate the model
score = model.evaluate(x_test, y_test)
print(f'Test loss: {score[0]}, Test accuracy: {score[1]}')