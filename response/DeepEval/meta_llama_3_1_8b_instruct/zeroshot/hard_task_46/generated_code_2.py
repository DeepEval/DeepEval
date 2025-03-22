import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the model
    inputs = keras.Input(shape=input_shape)

    # First block
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    branch1 = layers.SeparableConv2D(32, (1, 1), activation='relu')(x[0])
    branch2 = layers.SeparableConv2D(32, (3, 3), activation='relu')(x[1])
    branch3 = layers.SeparableConv2D(32, (5, 5), activation='relu')(x[2])
    x = layers.Concatenate()([branch1, branch2, branch3])

    # Second block
    x = layers.SeparableConv2D(32, (3, 3), activation='relu')(x)
    x = layers.SeparableConv2D(32, (1, 1), activation='relu')(x)
    x = layers.SeparableConv2D(32, (3, 3), activation='relu')(x)
    x_pool = layers.MaxPooling2D((2, 2))(x)
    x = layers.Concatenate()([x, x_pool])

    # Global average pooling and fully connected layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs, outputs)

    return model

# Compile the model
def compile_model(model):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, train_data, train_labels, epochs):
    model.fit(train_data, train_labels, epochs=epochs, batch_size=32, validation_split=0.2)
    return model

# Load the CIFAR-10 dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test

# Main function
if __name__ == "__main__":
    # Load the CIFAR-10 dataset
    x_train, y_train, x_test, y_test = load_data()

    # Create the model
    model = dl_model()

    # Compile the model
    model = compile_model(model)

    # Train the model
    model = train_model(model, x_train, y_train, epochs=10)

    # Evaluate the model
    model.evaluate(x_test, y_test)