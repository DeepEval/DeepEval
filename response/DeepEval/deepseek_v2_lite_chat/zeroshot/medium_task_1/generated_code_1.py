from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model building
def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer 1
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    # MaxPooling layer
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Convolutional layer 2
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    # MaxPooling layer
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Flatten layer
    flat = Flatten()(pool2)
    
    # Fully connected layer 1
    dense1 = Dense(512, activation='relu')(flat)
    # Fully connected layer 2
    dense2 = Dense(10, activation='softmax')(dense1)
    
    # Connect the layers
    model = Model(inputs=input_layer, outputs=dense2)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Display the model summary
model.summary()

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Predict using the model
predictions = model.predict(x_test)

# Convert predictions to class labels
predicted_classes = [np.argmax(prediction) for prediction in predictions]

# Print some predictions
print('Predictions:', predicted_classes[:10])