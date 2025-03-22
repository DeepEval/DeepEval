import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.layers import Layer

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Function to define the model
def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    channel_splits = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Feature extraction through convolutional layers
    conv1 = Conv2D(64, (1, 1), activation='relu')(channel_splits[0])
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    conv3 = Conv2D(64, (5, 5), activation='relu')(conv2)
    
    # Apply dropout to reduce overfitting
    conv1 = tf.keras.layers.Dropout(0.2)(conv1)
    conv2 = tf.keras.layers.Dropout(0.2)(conv2)
    conv3 = tf.keras.layers.Dropout(0.2)(conv3)
    
    # Concatenate the outputs from the three groups
    concat = concatenate([conv1, conv2, conv3])
    
    # Second block
    branch1 = Conv2D(128, (1, 1), activation='relu')(channel_splits[1])
    branch2 = Conv2D(128, (3, 3), activation='relu')(branch1)
    branch3 = Conv2D(128, (5, 5), activation='relu')(branch2)
    
    branch4 = AveragePooling2D(pool_size=(3, 3))(channel_splits[2])
    branch4 = Conv2D(128, (1, 1), activation='relu')(branch4)
    
    # Concatenate the outputs from all branches
    concat_branch = concatenate([branch1, branch2, branch3, branch4])
    
    # Flatten and fully connected layers
    flatten = Flatten()(concat_branch)
    fc1 = Dense(512, activation='relu')(flatten)
    output = Dense(10, activation='softmax')(fc1)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Build the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model (example code, replace with your actual training loop)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Save the model
model.save('cifar10_model.h5')