import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Convolutional layers with batch normalization and ReLU activation
    def block1(x):
        x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    # Block 2: Convolutional layers with batch normalization and ReLU activation
    def block2(x):
        x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    # Block 3: Convolutional layers with batch normalization and ReLU activation
    def block3(x):
        x = Conv2D(filters=128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    # Concatenate outputs from each block
    x1 = block1(input_layer)
    x2 = block2(input_layer)
    x3 = block3(input_layer)
    
    # Concatenate along the channel dimension
    concat = Concatenate(axis=-1)([x1, x2, x3])
    
    # Flatten and pass through two fully connected layers
    x = Flatten()(concat)
    x = Dense(units=1024, activation='relu')(x)
    output = Dense(units=10, activation='softmax')(x)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')