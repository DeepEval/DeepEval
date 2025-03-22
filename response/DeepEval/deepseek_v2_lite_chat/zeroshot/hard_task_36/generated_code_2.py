import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the main pathway
def main_pathway(input_shape):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.5)(x)
    
    # Branch pathway
    branch_x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Concatenate outputs from both pathways
    concatenated = Concatenate()([x, branch_x])
    
    # Flatten and fully connected layers
    x = Flatten()(concatenated)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)  # Assuming 10 classes for MNIST
    
    return Model(inputs=input_shape, outputs=output)

# Construct the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model on test data
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')