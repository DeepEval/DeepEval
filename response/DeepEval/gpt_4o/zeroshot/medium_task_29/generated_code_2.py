import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # MaxPooling layers with different pool sizes and stride
    pool_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x)
    pool_2x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    pool_4x4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x)
    
    # Flatten each pooled output
    flat_1x1 = Flatten()(pool_1x1)
    flat_2x2 = Flatten()(pool_2x2)
    flat_4x4 = Flatten()(pool_4x4)
    
    # Concatenate flattened outputs
    concatenated = Concatenate()([flat_1x1, flat_2x2, flat_4x4])
    
    # Fully connected layers
    dense1 = Dense(256, activation='relu')(concatenated)
    output_layer = Dense(10, activation='softmax')(dense1)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Summary of the model
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)