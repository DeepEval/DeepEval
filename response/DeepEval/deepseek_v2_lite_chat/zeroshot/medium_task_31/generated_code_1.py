import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create input shape
input_shape = (32, 32, 3)


def dl_model():
    # Lambda layer to split the input into three groups
    split = Lambda(lambda x: tf.split(x, 3, axis=1), arguments={'num_split': 3})
    
    # Convolution layers with different kernel sizes
    conv1 = Conv2D(32, (1, 1), activation='relu', padding='same')(split[0])
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(split[1])
    conv3 = Conv2D(64, (5, 5), activation='relu', padding='same')(split[2])
    
    # Concatenate feature maps
    x = concatenate([conv1, conv2, conv3])
    
    # Flatten the concatenated features
    x = Flatten()(x)
    
    # Fully connected layers for classification
    output = Dense(1024, activation='relu')(x)
    output = Dense(512, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)
    
    # Create the model
    model = Model(inputs=Input(shape=input_shape), outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Print the model summary
model.summary()