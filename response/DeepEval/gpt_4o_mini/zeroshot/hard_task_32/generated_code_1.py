import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def depthwise_separable_block(inputs, filters):
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    return x

def dl_model():
    inputs = layers.Input(shape=(28, 28, 1))

    # Branch 1
    branch1 = depthwise_separable_block(inputs, filters=32)
    branch1 = layers.MaxPooling2D(pool_size=(2, 2))(branch1)

    # Branch 2
    branch2 = depthwise_separable_block(inputs, filters=64)
    branch2 = layers.MaxPooling2D(pool_size=(2, 2))(branch2)

    # Branch 3
    branch3 = depthwise_separable_block(inputs, filters=128)
    branch3 = layers.MaxPooling2D(pool_size=(2, 2))(branch3)

    # Concatenate branches
    concatenated = layers.concatenate([branch1, branch2, branch3])

    # Fully connected layers
    x = layers.Flatten()(concatenated)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Train the model (optional)
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

# Evaluate the model (optional)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')