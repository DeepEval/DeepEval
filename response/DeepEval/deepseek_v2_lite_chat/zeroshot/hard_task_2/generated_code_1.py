import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data for the model
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the Lambda layer for splitting the input into three groups
lambda_layer = Lambda(lambda x: tf.split(x, 3, axis=3))([tf.split(tf.expand_dims(x_train, 3), 3, axis=3)])

# Add Lambda layers to each group
x_train = [lambda_layer[i] for i in range(3)]

# Define the model architecture
def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First 1x1 convolution
    conv1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    # Second 3x3 convolution
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    # Third 1x1 convolution
    conv3 = Conv2D(64, (3, 3), activation='relu')(conv2)
    
    # Split the output into three groups
    group1 = MaxPooling2D(pool_size=(2, 2))(conv3)
    group2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    group3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Concatenate the outputs of the three groups
    concat = concatenate([group1, group2, group3])
    
    # Add another 1x1 convolution for feature extraction
    conv4 = Conv2D(64, (1, 1), activation='relu')(concat)
    
    # Add an addition operation to combine with the original input
    main_path = tf.keras.layers.Add()([inputs, conv4])
    
    # Flatten and feed into a fully connected layer
    flat = Flatten()(main_path)
    dense = Dense(128, activation='relu')(flat)
    outputs = Dense(10, activation='softmax')(dense)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Print model summary
model.summary()

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)