import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=input_shape),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=5, validation_split=0.2)
    return model

def reset_model(model):
    # Get the trainable variables of the model
    trainable_vars = model.trainable_variables
    
    # Reset the weights of the model to their initial values
    for v in trainable_vars:
        if v.shape.rank == 0:  # Check if it's a scalar (needed for some initializers)
            v.assign(tf.zeros_like(v))  # Initialize to zero for Xavier/Glorot uniform initializer
        else:
            initial_value = v.numpy()  # For other initializers, save the initial value
            v.assign(tf.zeros_like(v))  # Reset to zero
    return model

# Generate some dummy data and labels
num_samples = 1000
num_features = 10
num_classes = 5
np.random.seed(123)
x_train = np.random.randn(num_samples, num_features)
y_train = np.random.randint(num_classes, size=(num_samples,))

# Build and train the model
model = build_model((num_features,), num_classes)
model = train_model(model, x_train, y_train)

# Reset the model
reset_model(model)
model = train_model(model, x_train, y_train)

# Call the method for validation
output = model.predict(x_train)

print("Model output:\n", output)