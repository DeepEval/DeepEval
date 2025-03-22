import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, add

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Function to create the model
def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    group1, group2, group3 = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    
    # Sequential model for group1
    x1 = Conv2D(16, (1, 1), padding='same')(group1)
    x1 = Conv2D(16, (3, 3), padding='same')(x1)
    x1 = MaxPooling2D()(x1)
    x1 = tf.keras.layers.Dropout(0.2)(x1)
    
    # Sequential model for group2
    x2 = Conv2D(32, (1, 1), padding='same')(group2)
    x2 = Conv2D(32, (3, 3), padding='same')(x2)
    x2 = MaxPooling2D()(x2)
    x2 = tf.keras.layers.Dropout(0.2)(x2)
    
    # Sequential model for group3
    x3 = Conv2D(64, (1, 1), padding='same')(group3)
    x3 = Conv2D(64, (3, 3), padding='same')(x3)
    x3 = MaxPooling2D()(x3)
    x3 = tf.keras.layers.Dropout(0.2)(x3)
    
    # 1x1 convolution for branch pathway
    x4 = Conv2D(64, (1, 1), padding='same')(inputs)
    
    # Concatenate the outputs from the main pathway and branch pathway
    concat = concatenate([x1, x2, x3, x4])
    
    # Flatten and pass through a fully connected layer
    x = Flatten()(concat)
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Display the model summary
model.summary()

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)