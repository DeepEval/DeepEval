import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_img = layers.Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_channels = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_img)
    
    # Define different convolutional layers for each channel
    conv1x1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(split_channels[0])
    conv3x3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(split_channels[1])
    conv5x5 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(split_channels[2])
    
    # Concatenate the outputs from the three convolution layers
    concatenated = layers.concatenate([conv1x1, conv3x3, conv5x5], axis=-1)
    
    # Flatten the concatenated features
    flattened = layers.Flatten()(concatenated)
    
    # Fully connected layers for classification
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    
    # Output layer with softmax activation for multi-class classification
    output = layers.Dense(10, activation='softmax')(dense2)
    
    # Construct the model
    model = models.Model(inputs=input_img, outputs=output)
    
    return model

# Optional: Load CIFAR-10 dataset and prepare data (if needed)
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# Example of creating the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 dataset (if you wish to use it)
(x_train, y_train), (x_test, y_test) = load_cifar10()

# Train the model (uncomment to train)
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))