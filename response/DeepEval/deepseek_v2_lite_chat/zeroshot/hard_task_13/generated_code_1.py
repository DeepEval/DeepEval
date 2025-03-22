import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


def dl_model():
    # Input shape for CIFAR-10
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Define the first branch with a 1x1 convolution
    branch1 = Conv2D(64, (1, 1), activation='relu')(inputs)
    
    # Define the second branch with a 3x3 convolution
    branch2 = Conv2D(64, (3, 3), activation='relu')(inputs)
    
    # Define the third branch with a 5x5 convolution
    branch3 = Conv2D(64, (5, 5), activation='relu')(inputs)
    
    # Define the fourth branch with a 3x3 max pooling
    branch4 = MaxPooling2D()(inputs)
    
    # Concatenate the outputs of the four branches
    x = concatenate([branch1, branch2, branch3, branch4])
    
    # Apply global average pooling to reduce dimensionality
    x = GlobalAveragePooling2D()(x)
    
    # Output layer with fully connected layers
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))