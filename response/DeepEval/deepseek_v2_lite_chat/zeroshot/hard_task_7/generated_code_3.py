import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten, Reshape, Permute

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def dl_model():
    # Input shape
    input_shape = (28, 28, 1)

    # Block 1: Initial Convolution and Split
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (1, 1), padding='same')(input_layer)
    x = Conv2D(32, (1, 1), use_depthwise=True)(x)
    x = Conv2D(64, (1, 1))(x)
    
    # Block 2: Split and Merge
    split = Lambda(lambda x: tf.split(x, 2, axis=-1))(x)
    first_group = split[0]
    second_group = split[1]
    
    # Operations on first group
    x = Conv2D(64, (1, 1))(first_group)
    x = Conv2D(128, (1, 1), use_depthwise=True)(x)
    x = Conv2D(256, (1, 1))(x)
    
    # Operations on second group
    y = second_group  # Passed through without modification
    
    # Merge
    merged = tf.concat([x, y], axis=-1)
    
    # Block 2: Reshape, Shuffle, Flatten, and Output
    shape = tf.shape(merged)
    batch_size = shape[0]
    height, width, channels = shape[1], shape[2], shape[3]
    
    # Reshape into four groups
    groups = channels // 2
    reshaped = Reshape((height, width, groups, 2))(merged)
    
    # Swap third and fourth dimensions
    shuffled = Permute((2, 3, 1))(reshaped)
    
    # Reshape back to original shape
    output = Reshape((batch_size * height * width, 2))(shuffled)
    
    # Fully connected layer
    output_layer = Dense(10, activation='softmax')(output)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)