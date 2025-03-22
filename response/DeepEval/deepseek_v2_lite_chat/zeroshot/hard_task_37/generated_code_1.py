from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense
from keras.layers import BatchNormalization, Activation

import keras.backend as K


def dl_model():
    # Define input shape
    input_shape = (28, 28, 1)  # MNIST images are 28x28 and grayscale
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Branch 1: Convolutional layers
    branch1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    branch1 = BatchNormalization()(branch1)
    branch1 = Conv2D(64, (3, 3), activation='relu')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2))(branch1)
    
    # Branch 2: Convolutional layers
    branch2 = Conv2D(32, (3, 3), activation='relu')(inputs)
    branch2 = BatchNormalization()(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2))(branch2)
    
    # Branch 3: Direct connection from input to convolutional layer
    branch3 = Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Combine outputs from different branches
    combined = Concatenate(axis=-1)([branch1, branch2, branch3])
    
    # Flatten and fully connected layers
    flat = Flatten()(combined)
    output = Dense(10, activation='softmax')(flat)  # Assuming 10 classes for MNIST
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    return model


# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Assuming you have the MNIST dataset split into training and validation sets
# X_train, X_test, y_train, y_test = load_mnist_data()

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)