from tensorflow.keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 with 1 channel
    num_classes = 10  # MNIST has 10 classes

    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Input layer
    inputs = Input(shape=input_shape)

    # First block: main path and branch path
    # Main path
    x_main = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x_main = Conv2D(1, (3, 3), activation='relu', padding='same')(x_main)

    # Branch path (identity mapping)
    x_branch = inputs

    # Combine paths with addition
    x_combined = Add()([x_main, x_branch])

    # Second block: three max pooling layers
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x_combined)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x_combined)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x_combined)

    # Flatten and concatenate
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    concatenated = Concatenate()([flat1, flat2, flat3])

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(concatenated)
    fc2 = Dense(64, activation='relu')(fc1)
    outputs = Dense(num_classes, activation='softmax')(fc2)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to compile and train the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)