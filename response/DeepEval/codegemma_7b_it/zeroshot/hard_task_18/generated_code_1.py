from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # First block
    conv1_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(inputs)
    conv1_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1_1)
    avg_pool1 = layers.AveragePooling2D()(conv1_2)

    # Second block
    conv2_1 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(avg_pool1)
    conv2_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv2_1)
    avg_pool2 = layers.AveragePooling2D()(conv2_2)

    # Main path
    flatten = layers.Flatten()(avg_pool2)
    dense1 = layers.Dense(units=128, activation='relu')(flatten)
    dense2 = layers.Dense(units=10, activation='softmax')(dense1)

    # Combine outputs
    output = layers.add([dense2, flatten])

    # Model definition
    model = models.Model(inputs=inputs, outputs=output)

    # Compilation
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model