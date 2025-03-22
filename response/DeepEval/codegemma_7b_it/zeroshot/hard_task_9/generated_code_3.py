import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Create input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Main path
    x1 = layers.Conv2D(64, (1, 1), padding='same')(inputs)
    x2 = layers.Conv2D(64, (1, 1), padding='same')(inputs)
    x2 = layers.Conv2D(64, (3, 3), padding='same')(x2)
    x3 = layers.Conv2D(64, (1, 1), padding='same')(inputs)
    x3 = layers.Conv2D(64, (3, 3), padding='same')(x3)
    x3 = layers.Conv2D(64, (3, 3), padding='same')(x3)
    concat = layers.Concatenate()([x1, x2, x3])
    x = layers.Conv2D(3, (1, 1), padding='same')(concat)
    x = layers.Add()([x, inputs])

    # Branch
    y = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    y = layers.Conv2D(64, (3, 3), padding='same')(y)

    # Classification
    y = layers.GlobalAveragePooling2D()(y)
    outputs = layers.Dense(10, activation='softmax')(y)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(x_train, y_train, epochs=10)

    # Evaluate model
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Test accuracy:', accuracy)

    return model