from tensorflow.keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for MNIST images, which are 28x28 pixels with a single channel
    inputs = Input(shape=(28, 28, 1))
    
    # Average pooling layer with a 5x5 window and a stride of 3x3
    x = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(inputs)
    
    # 1x1 convolutional layer to enhance depth and introduce nonlinearity
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    
    # Flatten the feature maps
    x = Flatten()(x)
    
    # First fully connected layer
    x = Dense(units=128, activation='relu')(x)
    
    # Dropout layer to mitigate overfitting
    x = Dropout(rate=0.5)(x)
    
    # Second fully connected layer
    x = Dense(units=64, activation='relu')(x)
    
    # Output layer for 10 classes with softmax activation
    outputs = Dense(units=10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare data for demonstration purposes
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Print the model summary
model.summary()