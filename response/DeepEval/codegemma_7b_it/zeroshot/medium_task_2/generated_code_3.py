from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define input layer
    input_img = layers.Input(shape=(32, 32, 3))

    # Main path
    tower_1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
    tower_1 = layers.Conv2D(64, (3, 3), activation='relu')(tower_1)
    tower_1 = layers.MaxPooling2D(pool_size=(2, 2))(tower_1)

    # Branch path
    tower_2 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(input_img)

    # Combine features
    merged = layers.concatenate([tower_1, tower_2])

    # Fully connected layers
    merged = layers.Flatten()(merged)
    merged = layers.Dense(512, activation='relu')(merged)
    merged = layers.Dense(10, activation='softmax')(merged)

    # Create model
    model = models.Model(inputs=input_img, outputs=merged)

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
score = model.evaluate(x_test, y_test)

print('Test accuracy:', score[1])