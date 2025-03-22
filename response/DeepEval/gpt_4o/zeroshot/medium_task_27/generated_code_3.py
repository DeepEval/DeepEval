from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)

    # Define the input
    inputs = Input(shape=input_shape)

    # Branch 1 - 3x3 Conv
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)

    # Branch 2 - 5x5 Conv
    x2 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same')(x2)

    # Combine branches by addition
    combined = Add()([x1, x2])

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(combined)

    # Attention mechanism
    attention_weights = Dense(10, activation='softmax')(gap)

    # Weighted branch outputs
    weighted_x1 = Multiply()([x1, attention_weights])
    weighted_x2 = Multiply()([x2, attention_weights])

    # Final weighted output
    final_output = Add()([weighted_x1, weighted_x2])

    # Global Average Pooling for final output
    final_gap = GlobalAveragePooling2D()(final_output)

    # Output layer
    outputs = Dense(10, activation='softmax')(final_gap)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Summary of the model
model.summary()

# You can train the model using:
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))