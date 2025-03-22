import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.utils import to_categorical
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

def dl_model():
    # Load and preprocess CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Number of classes
    num_classes = y_train.shape[1]
    
    # Input layer
    inputs = Input(shape=x_train.shape[1:])
    
    # Path 1: Main path
    x = inputs
    for _ in range(2):
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D()(x)
    
    # Path 2: Branch path
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    
    # Concatenate paths
    x = Concatenate()([x, x])
    
    # Additional convolution and flattening
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and train the model
model = dl_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=128)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

# Example of showing the first image from the test set
index = 0
plt.imshow(x_test[index])
plt.show()

print('Label:', np.argmax(y_test[index]))