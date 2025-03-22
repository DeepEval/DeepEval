from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Branch path: Passes through the input layer directly
    branch_inputs = inputs
    
    # Main path
    x = layers.Conv2D(64, kernel_size=3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    
    # Reshape weights for element-wise multiplication
    weights = layers.Dense(x.shape[-1])(x)
    weights = layers.Reshape(x.shape[:-1])(weights)
    
    # Element-wise multiplication
    x = x * weights
    
    # Combine outputs from the main path and the branch path
    combined = layers.add([x, branch_inputs])
    
    # Final fully connected layers
    x = layers.Dense(10, activation="softmax")(combined)
    
    # Model architecture
    model = keras.Model(inputs=[inputs, branch_inputs], outputs=x)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and compile the model
model = dl_model()
model.summary()
model.fit([x_train, x_train], y_train, epochs=10, validation_data=([x_test, x_test], y_test))