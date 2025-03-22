from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    # First block of Conv + MaxPooling
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second block of Conv + MaxPooling
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Branch Path
    # One block of Conv + MaxPooling
    y = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    y = MaxPooling2D(pool_size=(2, 2))(y)

    # Combine outputs from both paths using an addition operation
    combined = Add()([x, y])

    # Flatten the combined output
    flattened = Flatten()(combined)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    output_layer = Dense(10, activation='softmax')(fc1)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Loading CIFAR-10 dataset (for demonstration purposes)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Instantiate the model
model = dl_model()

# Print the model summary
model.summary()