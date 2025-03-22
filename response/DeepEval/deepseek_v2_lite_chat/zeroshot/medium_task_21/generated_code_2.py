import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model architecture
def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Define the four branches
    branch1 = Conv2D(64, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(64, (1, 1))(inputs)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch3 = Conv2D(64, (1, 1))(branch2)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)
    branch4 = MaxPooling2D((3, 3))(inputs)
    branch4 = Conv2D(64, (1, 1))(branch4)
    
    # Apply dropout to each branch
    branch1 = Dropout(0.5)(branch1)
    branch2 = Dropout(0.5)(branch2)
    branch3 = Dropout(0.5)(branch3)
    branch4 = Dropout(0.5)(branch4)
    
    # Concatenate the outputs
    x = concatenate([branch1, branch2, branch3, branch4])
    
    # Flatten and pass through three fully connected layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create and compile the model
model = dl_model()
model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Return the model
return model