import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, concatenate
from keras.layers import Lambda

# Set the number of classes for CIFAR-10
num_classes = 10

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch with 1x1 convolution
    branch1 = Conv2D(filters=64, kernel_size=1, activation='relu')(input_layer)
    
    # Branch with 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=1)(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=3, padding='same')(branch2)
    
    # Branch with 1x1 convolution followed by two 3x3 convolutions
    branch3 = Conv2D(filters=64, kernel_size=1)(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=3)(branch3)
    branch3 = Conv2D(filters=64, kernel_size=3, padding='same')(branch3)
    
    # Concatenate the outputs of the three branches
    concat = concatenate([branch1, branch2, branch3])
    
    # 1x1 convolution to adjust dimensions
    conv1x1 = Conv2D(filters=64, kernel_size=1)(concat)
    
    # Add the main path to the concatenated output
    add = Add()([input_layer, conv1x1])
    
    # Main path to the classification head
    model = Model(inputs=input_layer, outputs=add)
    
    # Add a classification head
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to categorical one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)