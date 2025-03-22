from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1
    # Block 1
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)
    
    # Block 2
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)

    # Path 2
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    
    # Combine paths with addition
    combined = Add()([x1, x2])
    
    # Flatten the combined features
    flattened = Flatten()(combined)
    
    # Fully connected layer to map to class probabilities
    output_layer = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
# model = dl_model()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))