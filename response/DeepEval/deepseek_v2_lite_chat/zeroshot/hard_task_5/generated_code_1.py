import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model parameters
input_shape = (32, 32, 3)  # Size of each input image
num_classes = y_train.shape[1]  # Number of classes

def dl_model():
    # Define the main path
    def block1():
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.split(inputs, 3, axis=2)
        x = [layers.Conv2D(32, (1, 1), activation='relu')(xi) for xi in x]
        x = tf.concat(x, axis=2)
        return tf.keras.Model(inputs, x)

    def block2(x):
        shape = x.shape[1:4]  # Shape before reshaping
        x = layers.Reshape(target_shape=(shape[0], shape[1], 3, shape[2] // 3))(x)
        x = layers.Permute([2, 3, 1])(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', strides=2)(x)
        return x

    def block3(x):
        x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(64, (1, 1), padding='same')(x)
        return x

    # Create the model
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = block1()(x)
    x = block2(x)
    x = block3(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()
model.summary()

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')