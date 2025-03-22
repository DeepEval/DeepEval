import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # Define input shape
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)
    
    # Block 1: Global Average Pooling, Fully Connected Layers, and Weighted Feature Output
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    weights = x
    weights = tf.reshape(weights, (-1, 32, 1, 1))
    weighted_features = Multiply()([inputs, weights])
    
    # Block 2: Two 3x3 Convolutional Layers, Max Pooling
    y = Conv2D(64, (3, 3), activation='relu', padding='same')(weighted_features)
    y = Conv2D(64, (3, 3), activation='relu', padding='same')(y)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(128, (3, 3), activation='relu', padding='same')(y)
    y = Conv2D(128, (3, 3), activation='relu', padding='same')(y)
    y = MaxPooling2D((2, 2))(y)
    
    # Branch from Block 1 connects directly to the output of Block 2
    branch = Dense(128, activation='relu')(weights)
    branch = Dense(128, activation='relu')(branch)
    
    # Fuse the outputs through addition
    combined = Add()([y, branch])
    
    # Add two fully connected layers for classification
    z = GlobalAveragePooling2D()(combined)
    z = Dense(128, activation='relu')(z)
    outputs = Dense(10, activation='softmax')(z)
    
    # Compile and return the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage
model = dl_model()
model.summary()