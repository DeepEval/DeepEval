import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.optimizers import Adam

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Set up the input shape for the model
input_shape = x_train[0].shape

def dl_model():
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Reshape the input tensor
    reshaped_inputs = Reshape((-1, input_shape[0]*input_shape[1]*input_shape[2]))(inputs)
    
    # Split into three groups
    group1, group2, group3 = tf.split(reshaped_inputs, 3, axis=-1)
    
    # Swap the third and fourth dimensions
    shuffled_inputs = tf.transpose(group1, perm=[0, 2, 3, 1])
    
    # Reshape back to (height, width, groups, channels_per_group)
    reshaped_shuffled = Reshape((input_shape[0], input_shape[1], input_shape[2], 3))(shuffled_inputs)
    
    # Combine the shuffled tensors
    combined_tensor = tf.concat([group1, group2, group3, reshaped_shuffled], axis=-1)
    
    # Convolutional layers
    conv1 = Conv2D(32, (3, 3), activation='relu')(combined_tensor)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Flatten and fully connected layer
    flat = Flatten()(pool)
    dense = Dense(512, activation='relu')(flat)
    
    # Output layer with softmax activation
    output = Dense(10, activation='softmax')(dense)
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Print model summary
model.summary()