import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the input shape
input_shape = (32, 32, 3)

# Function to create the model
def dl_model():
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Split input into three channel groups
    x = tf.split(inputs, 3, axis=-1)
    
    # Apply separable convolutions to each group
    y1 = Conv2D(32, (1, 1), activation='relu')(x[0])
    y2 = Conv2D(32, (3, 3), activation='relu')(x[1])
    y3 = Conv2D(32, (5, 5), activation='relu')(x[2])
    
    # Concatenate the outputs from separable convolutions
    z = Concatenate()([y1, y2, y3])
    
    # Additional fully connected layers
    z = Flatten()(z)
    z = Dense(512, activation='relu')(z)
    z = Dense(256, activation='relu')(z)
    z = Dense(10, activation='softmax')(z)  # Assuming 10 classes for CIFAR-10
    
    # Model
    model = Model(inputs=inputs, outputs=z)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()
model.summary()