from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First block
    x1 = Conv2D(32, (3, 3), padding='same', activation=None)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    # Second block
    x2 = Conv2D(64, (3, 3), padding='same', activation=None)(x1)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = Concatenate()([x1, x2])

    # Third block
    x3 = Conv2D(128, (3, 3), padding='same', activation=None)(x2)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x3 = Concatenate()([x2, x3])

    # Fully connected layers
    x = Flatten()(x3)
    x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=x)

    return model

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Instantiate the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# You can train the model using the following command:
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)