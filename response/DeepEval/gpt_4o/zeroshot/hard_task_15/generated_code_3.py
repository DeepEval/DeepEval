from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input shape based on CIFAR-10 images
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Main path
    # Applying Global Average Pooling
    x = GlobalAveragePooling2D()(inputs)
    # Two fully connected layers
    x = Dense(64, activation='relu')(x)
    x = Dense(input_shape[2], activation='relu')(x)
    # Reshape to match the input feature map dimensions
    x = Dense(input_shape[0] * input_shape[1] * input_shape[2], activation='relu')(x)
    x = Multiply()([inputs, x])

    # Branch path
    branch = inputs
    
    # Combine main and branch paths
    combined = Add()([x, branch])
    
    # Flatten the combined output
    combined = Flatten()(combined)
    
    # Fully connected layers to produce the final classification output
    combined = Dense(256, activation='relu')(combined)
    outputs = Dense(10, activation='softmax')(combined)  # CIFAR-10 has 10 classes
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))