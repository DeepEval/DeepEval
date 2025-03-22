from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocess data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Define input layer
    inputs = Input(shape=(32, 32, 3))

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)

    # Max-pooling layer
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Output features directly added with input layer
    outputs = x

    # Flatten and fully connected layers
    x = Flatten()(outputs)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

model = dl_model()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)

print('Loss:', loss)
print('Accuracy:', accuracy)