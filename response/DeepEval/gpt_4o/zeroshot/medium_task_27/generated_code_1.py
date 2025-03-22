import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First branch with 3x3 convolutions
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)

    # Second branch with 5x5 convolutions
    x2 = Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
    x2 = Conv2D(32, (5, 5), activation='relu', padding='same')(x2)

    # Combining the branches through addition
    combined = Add()([x1, x2])

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(combined)

    # Attention mechanism
    attention_dense1 = Dense(64, activation='relu')(gap)
    attention_weights = Dense(2, activation='softmax')(attention_dense1)

    # Splitting attention weights
    attention_weights1 = attention_weights[:, 0]
    attention_weights2 = attention_weights[:, 1]

    # Applying attention weights to branches
    weighted_x1 = Multiply()([x1, attention_weights1])
    weighted_x2 = Multiply()([x2, attention_weights2])

    # Adding weighted branches
    weighted_combined = Add()([weighted_x1, weighted_x2])

    # Final fully connected layer for classification
    output = Dense(10, activation='softmax')(GlobalAveragePooling2D()(weighted_combined))

    # Create and compile model
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
if __name__ == "__main__":
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Normalize data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Get the model
    model = dl_model()
    model.summary()

    # Optionally, you can fit the model with training data
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))