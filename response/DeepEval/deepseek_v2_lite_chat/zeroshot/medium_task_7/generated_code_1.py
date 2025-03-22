from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def dl_model():
    # Model architecture
    input_layer = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    
    # Concatenate the outputs of the first and second convolutional layers with the third
    concat = keras.layers.concatenate([x, x])
    
    # Process the concatenated output
    x = Conv2D(64, (3, 3), activation='relu')(concat)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    
    # Process the input directly
    x_input = Input(shape=(32, 32, 3))
    x_input_conv = Conv2D(64, (3, 3), activation='relu')(x_input)
    
    # Combine all paths
    x = keras.layers.concatenate([x, x_input_conv, concat])
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)  # Assuming 10 classes
    
    # Model architecture
    model = Model(inputs=[input_layer, x_input], outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create and train the model
model = dl_model()
model.fit([x_train, x_train], y_train, validation_data=([x_test, x_test], y_test), epochs=20)

# Evaluate the model
loss, accuracy = model.evaluate([x_test, x_test], y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')