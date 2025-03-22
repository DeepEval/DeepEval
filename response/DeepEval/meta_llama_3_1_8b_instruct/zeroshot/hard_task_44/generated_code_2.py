# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Define the model input
    inputs = keras.Input(shape=input_shape)
    
    # Block 1: Feature extraction with varying kernel sizes and concatenation
    x1 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    x1 = [layers.Conv2D(32, (1, 1), activation='relu')(x1[i]) for i in range(3)]
    x1 = layers.Concatenate()(x1)
    x1 = layers.Dropout(0.2)(x1)
    
    # Block 2: Four branches for feature fusion
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(x1)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(x1)
    branch3 = layers.Conv2D(32, (5, 5), activation='relu')(x1)
    branch4 = layers.Conv2D(32, (3, 3), activation='relu')(layers.MaxPooling2D((3, 3))(x1))
    
    # Concatenate the outputs from the four branches
    x2 = layers.Concatenate()([branch1, branch2, branch3, branch4])
    
    # Flatten and fully connected layer for classification
    x = layers.Flatten()(x2)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])