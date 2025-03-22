import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, BatchNormalization, Activation, Concatenate, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Define the function to create the model
def dl_model():
    # Define the input
    inputs = Input(shape=x_train.shape[1:])
    
    # Branch 1
    branch1 = Conv2D(32, 3, padding='same', use_bias=False, kernel_regularizer=l2(0.0005))(inputs)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)
    branch1 = Conv2D(32, 3, padding='same', use_bias=False)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)
    branch1 = Conv2D(64, 3, padding='same', use_bias=False)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)
    
    # Branch 2
    branch2 = Conv2D(32, 3, padding='same', use_bias=False)(inputs)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2D(32, 3, padding='same', use_bias=False)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    
    # Branch 3
    branch3 = Conv2D(32, 3, padding='same', use_bias=False)(inputs)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = Conv2D(32, 3, padding='same', use_bias=False)(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)
    
    # Dropout
    branch1 = tf.keras.layers.Dropout(0.5)(branch1)
    branch2 = tf.keras.layers.Dropout(0.5)(branch2)
    branch3 = tf.keras.layers.Dropout(0.5)(branch3)
    
    # Concatenate branches
    x = Concatenate()([branch1, branch2, branch3])
    
    # Fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=x)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Print model summary
model.summary()

# Train the model (this is a placeholder for actual training code)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Save the model (optional)
model.save('mnist_model.h5')