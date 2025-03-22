from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input
    inputs = Input(shape=(32, 32, 3))

    # Sequential blocks with three paths
    # First Path
    x1 = Conv2D(32, (3, 3), padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    # Second Path
    x2 = Conv2D(64, (3, 3), padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    # Third Path
    x3 = Conv2D(128, (3, 3), padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)

    # Parallel Path processing input directly
    parallel_path = Conv2D(32, (3, 3), padding='same')(inputs)
    parallel_path = BatchNormalization()(parallel_path)
    parallel_path = ReLU()(parallel_path)

    # Adding the outputs from all paths
    added_outputs = Add()([x1, x2, x3, parallel_path])

    # Flatten and pass through fully connected layers
    x = Flatten()(added_outputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=x)

    return model

# Example usage
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model summary
model.summary()