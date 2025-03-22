# Import necessary libraries
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# Load CIFAR-10 dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# Create the deep learning model
def dl_model():
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = load_data()

    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create the model
    inputs = layers.Input(shape=input_shape)
    
    # First convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    
    # Second convolutional layer
    y = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    
    # Third convolutional layer
    z = layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(y)
    
    # Concatenate the output of the convolutional layers along the channel dimension
    concat = layers.Concatenate()([x, y, z])
    
    # Flatten the concatenated output
    flat = layers.Flatten()(concat)
    
    # First fully connected layer
    fc1 = layers.Dense(128, activation='relu')(flat)
    
    # Dropout layer to prevent overfitting
    dropout1 = layers.Dropout(0.2)(fc1)
    
    # Second fully connected layer
    outputs = layers.Dense(10, activation='softmax')(dropout1)
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create and return the model
model = dl_model()