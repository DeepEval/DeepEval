import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.layers import Dropout

# Constants
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

def dl_model():
    # Input layer
    inputs = Input(shape=INPUT_SHAPE)
    
    # Define each branch
    def branch1(x):
        x = Conv2D(32, (1, 1), activation='relu')(x)
        x = Dropout(0.5)(x)
        return x
    
    def branch2(x):
        x = Conv2D(32, (1, 1))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Dropout(0.5)(x)
        return x
    
    def branch3(x):
        x = Conv2D(32, (1, 1))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.5)(x)
        return x
    
    def branch4(x):
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (1, 1))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Dropout(0.5)(x)
        return x
    
    # Process each branch
    x1 = branch1(inputs)
    x2 = branch2(inputs)
    x3 = branch3(inputs)
    x4 = branch4(inputs)
    
    # Concatenate all branches
    x = concatenate([x1, x2, x3, x4])
    
    # Flatten and pass through three FC layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Assuming cifar10 dataset is available
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

model = dl_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)