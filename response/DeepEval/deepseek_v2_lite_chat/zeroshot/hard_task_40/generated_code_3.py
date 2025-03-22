import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, concatenate, Dense, Reshape
from keras.layers import Dropout, Concatenate
from keras.datasets import mnist
from keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model
def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: pooling layers
    p1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(input_layer)
    p1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(p1)
    p2 = Conv2D(64, (2, 2), strides=(2, 2), padding='same')(input_layer)
    p2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(p2)
    p3 = Conv2D(64, (4, 4), strides=(4, 4), padding='same')(input_layer)
    p3 = MaxPooling2D(pool_size=(2, 2), strides=(4, 4), padding='same')(p3)
    
    # Flatten and concatenate
    flat_layer = Flatten()(concatenate([p1, p2, p3]))
    
    # Second block: fully connected layers
    dense1 = Dense(256, activation='relu')(flat_layer)
    reshape1 = Reshape((-1, 1))(dense1)
    
    # Paths for multi-scale feature extraction
    path1 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    path2 = Conv2D(64, (3, 3), activation='relu')(input_layer)
    path3 = Conv2D(64, (3, 3), activation='relu')(input_layer)
    path4 = Conv2D(64, (3, 3), activation='relu')(input_layer)
    path1 = MaxPooling2D(pool_size=(1, 1))(path1)
    path2 = MaxPooling2D(pool_size=(3, 3))(path2)
    path3 = MaxPooling2D(pool_size=(3, 3))(path3)
    path4 = MaxPooling2D(pool_size=(3, 3))(path4)
    
    # Dropout layers
    drop1 = Dropout(0.5)(path1)
    drop2 = Dropout(0.5)(path2)
    drop3 = Dropout(0.5)(path3)
    drop4 = Dropout(0.5)(path4)
    
    # Output layers
    dense2 = Dense(128, activation='relu')(drop4)
    dense3 = Dense(64, activation='relu')(dense2)
    output = Dense(10, activation='softmax')(dense3)
    
    # Model
    model = Model(inputs=input_layer, outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()
model.summary()

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')