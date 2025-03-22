import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Function to create the model
def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Define the basic block
    def basic_block(input_layer):
        # Conv2D layer
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        norm1 = BatchNormalization()(conv1)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(norm1)
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
        
        return norm1, pool1
    
    # Define the main model structure
    def main_structure(input_layer):
        block1 = basic_block(input_layer)
        block2 = basic_block(block1[0])
        
        # Add a layer to reduce dimensionality
        flat1 = Flatten()(block2[1])
        
        # Connect the branches
        add1 = Add()([block1[1], flat1])
        
        # Additional layers for feature enhancement
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(add1)
        norm2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(norm2)
        
        # Average pooling
        avg_pool = AveragePooling2D(pool_size=(2, 2))(conv2)
        
        # Flatten and pass through a fully connected layer
        flat2 = Flatten()(avg_pool)
        dense1 = Dense(256, activation='relu')(flat2)
        
        # Output layer
        output = Dense(10, activation='softmax')(dense1)
        
        # Create the model
        model = Model(inputs=input_layer, outputs=output)
        
        return model
    
    # Return the main structure of the model
    return main_structure(input_layer)

# Create the model
model = dl_model()

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)