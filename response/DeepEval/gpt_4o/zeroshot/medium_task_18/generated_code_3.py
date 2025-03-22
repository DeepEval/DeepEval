from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Define the input shape based on the CIFAR-10 dataset (32x32 RGB images)
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)

    # 1x1 Convolution
    conv1x1 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)

    # 3x3 Convolution
    conv3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

    # 5x5 Convolution
    conv5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)

    # 3x3 Max Pooling
    maxpool3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)

    # Concatenate all the feature maps
    concatenated = Concatenate()([conv1x1, conv3x3, conv5x5, maxpool3x3])

    # Flatten the concatenated outputs
    flat = Flatten()(concatenated)

    # Fully connected layer with 128 neurons
    fc1 = Dense(128, activation='relu')(flat)

    # Fully connected layer with 10 neurons (CIFAR-10 has 10 classes)
    output = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=inputs, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage: 
# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Print the model summary
model.summary()

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy}")