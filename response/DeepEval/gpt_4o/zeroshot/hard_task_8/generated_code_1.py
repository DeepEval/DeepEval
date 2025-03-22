import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 with a single channel
    num_classes = 10  # There are 10 classes for MNIST digits

    inputs = Input(shape=input_shape)

    # Block 1 - Primary Path
    x1 = Conv2D(16, (1, 1), activation='relu', padding='same')(inputs)
    x1 = DepthwiseConv2D((3, 3), activation='relu', padding='same')(x1)
    x1 = Conv2D(32, (1, 1), activation='relu', padding='same')(x1)

    # Block 1 - Branch Path
    x2 = DepthwiseConv2D((3, 3), activation='relu', padding='same')(inputs)
    x2 = Conv2D(32, (1, 1), activation='relu', padding='same')(x2)

    # Concatenate features from both paths
    x = Concatenate(axis=-1)([x1, x2])

    # Block 2 - Channel Shuffling
    # Get the shape of the features
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]
    groups = 4
    channels_per_group = channels // groups

    # Reshape, Permute, and Reshape back
    x = Reshape((height, width, groups, channels_per_group))(x)
    x = Permute((1, 2, 4, 3))(x)
    x = Reshape((height, width, channels))(x)

    # Final output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

# Load MNIST data and preprocess
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Expand dimensions to include channel
x_train = x_train[..., None]
x_test = x_test[..., None]

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)