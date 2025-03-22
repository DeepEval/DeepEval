from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input shape based on the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Input Layer
    inputs = Input(shape=input_shape)

    # Block 1
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(inputs)
    # Fully Connected Layer 1
    fc1 = Dense(input_shape[2], activation='relu')(gap)
    # Fully Connected Layer 2
    fc2 = Dense(input_shape[2], activation='sigmoid')(fc1)
    # Reshape to match input shape
    weights = Reshape((1, 1, input_shape[2]))(fc2)
    # Multiply weights with input
    weighted_features = Multiply()([inputs, weights])

    # Block 2
    # First 3x3 Convolutional Layer
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(weighted_features)
    # Second 3x3 Convolutional Layer
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    # Max Pooling Layer
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Add the output of Block 1 to the output of Block 2
    added = Add()([max_pool, weighted_features])

    # Flatten the combined output
    flatten = Flatten()(added)
    # Fully Connected Layer 3
    fc3 = Dense(128, activation='relu')(flatten)
    # Fully Connected Layer 4 (Output Layer)
    outputs = Dense(10, activation='softmax')(fc3)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = dl_model()
model.summary()
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))