import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data to include the channel dimension (for Conv2D)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Block 1: Convolutional layers
    x = Conv2D(32, (1, 1), padding='same')(inputs)
    x = Conv2D(32, (1, 1), strides=1, padding='same')(x)
    x = DepthwiseConv2D((3, 3), strides=1, padding='same')(x)
    x = Conv2D(64, (1, 1), padding='same')(x)
    
    # Split the output into two groups
    group1, group2 = tf.split(x, 2, axis=-1)
    
    # Block 2: Pass group2 through unchanged, modify group1
    group1 = Conv2D(64, (1, 1), padding='same')(group1)
    group1 = Conv2D(128, (1, 1), padding='same')(group1)
    
    # Concatenate the outputs of Block 1
    concat = concatenate([group2, group1])
    
    # Block 3: Channel shuffling and reshaping
    shape = tf.shape(concat)
    batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    new_channels = channels // 4
    new_height = new_channels // 2 * 3
    new_width = height // 4
    
    # Reshape to four groups, then swap the third and fourth dimensions
    reshaped = tf.reshape(concat, (batch, new_height, new_width, 4, new_channels))
    reshaped = tf.transpose(reshaped, (0, 1, 3, 4, 2))
    
    # Reshape back to the original shape
    reshaped = tf.reshape(reshaped, (batch, height, width, channels))
    
    # Flatten and pass through a fully connected layer
    flattened = Flatten()(reshaped)
    outputs = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model