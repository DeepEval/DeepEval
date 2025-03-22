import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from keras.layers import Concatenate, AveragePooling2D

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the data for the model
input_shape = (32, 32, 3)  # Size of the images in the dataset
x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

# Function to define the model
def dl_model():
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Four feature extraction branches
    branch1 = Conv2D(16, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(16, (1, 1))(inputs)
    branch2 = Conv2D(16, (3, 3), activation='relu')(branch2)
    branch3 = Conv2D(16, (1, 1))(branch3)
    branch3 = Conv2D(16, (3, 3), activation='relu')(branch3)
    branch4 = AveragePooling2D()(inputs)
    branch4 = Conv2D(16, (1, 1))(branch4)
    
    # Apply dropout
    branch1 = Dropout(0.5)(branch1)
    branch2 = Dropout(0.5)(branch2)
    branch3 = Dropout(0.5)(branch3)
    branch4 = Dropout(0.5)(branch4)
    
    # Concatenate the outputs from all branches
    x = concatenate([branch1, branch2, branch3, branch4])
    
    # Flatten and pass through fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Build the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)