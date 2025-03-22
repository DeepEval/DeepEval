import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dense, Flatten, Lambda, GlobalAveragePooling2D
from tensorflow.keras.layers import Layer, Concatenate

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Parameters
input_shape = (32, 32, 3)  # Input image shape
num_classes = 10  # Number of output classes

# Helper function to create a model block
def block(inputs, filters, size, strides=(1, 1), padding='same', activation='relu'):
    x = Conv2D(filters=filters, kernel_size=size, strides=strides, padding=padding)(inputs)
    x = Conv2D(filters=filters, kernel_size=size, activation=activation)(x)
    return x

# Function to create the model
def dl_model():
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    x1 = block(x[0], 32, 3)
    x2 = block(x[1], 64, 3, strides=(2, 2))
    x3 = block(x[2], 32, 3)
    x = Concatenate()([x1, x2, x3])
    
    # Transition Convolution
    x = Conv2D(512, 3, padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 2
    x = block(x, 512, 3)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    
    # Generate channel-matching weights
    dim = int(x.shape[-1])
    x = Dense(dim * dim)(x)
    weights = Reshape((dim, dim, dim))(x)
    
    # Multiply and add
    output = Multiply()([x, weights])
    output = Conv2D(num_classes, 1, activation='softmax')(output)
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()