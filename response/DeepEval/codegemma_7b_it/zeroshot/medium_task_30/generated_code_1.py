import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define model architecture
    inputs = keras.Input(shape=(32, 32, 3))

    # Three average pooling layers with different window sizes and strides
    avg_pool_1 = layers.AveragePooling2D(pool_size=1, strides=1)(inputs)
    avg_pool_2 = layers.AveragePooling2D(pool_size=2, strides=2)(inputs)
    avg_pool_4 = layers.AveragePooling2D(pool_size=4, strides=4)(inputs)

    # Flatten outputs of pooling layers
    avg_pool_1 = layers.Flatten()(avg_pool_1)
    avg_pool_2 = layers.Flatten()(avg_pool_2)
    avg_pool_4 = layers.Flatten()(avg_pool_4)

    # Concatenate flattened features
    concat_features = layers.concatenate([avg_pool_1, avg_pool_2, avg_pool_4])

    # Two fully connected layers
    dense_layer_1 = layers.Dense(64, activation='relu')(concat_features)
    dense_layer_2 = layers.Dense(10, activation='softmax')(dense_layer_1)

    # Create model
    model = keras.Model(inputs=inputs, outputs=dense_layer_2)

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Build and train model
model = dl_model()
model.fit(x_train, y_train, epochs=10)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', accuracy)