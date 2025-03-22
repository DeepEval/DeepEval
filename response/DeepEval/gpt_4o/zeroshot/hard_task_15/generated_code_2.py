from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    # Global Average Pooling
    x = GlobalAveragePooling2D()(input_layer)
    
    # Two fully connected layers
    x = Dense(units=256, activation='relu')(x)
    x = Dense(units=3, activation='sigmoid')(x)  # Assuming 3 channels for simplicity

    # Reshape and multiply element-wise
    x = Multiply()([input_layer, x])

    # Branch path
    branch = input_layer

    # Add the outputs of the main path and branch path
    combined = Add()([x, branch])

    # Flatten the combined features
    combined = Flatten()(combined)

    # Final fully connected layers to produce the final probability distribution
    combined = Dense(units=512, activation='relu')(combined)
    output_layer = Dense(units=10, activation='softmax')(combined)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize and preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))