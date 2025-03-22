from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():

    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # First block
    x = inputs
    for scale in [1, 2, 4]:
        x = layers.MaxPooling2D(pool_size=(scale, scale), strides=(scale, scale), padding='same')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)

    # Second block
    x = layers.Dense(units=512, activation='relu')(x)
    x = layers.Reshape((1, 1, 512))(x)
    x = layers.Lambda(tf.split, arguments={'axis': -1, 'num_or_size_splits': 4})(x)

    for kernel_size in [1, 3, 5, 7]:
        y = layers.SeparableConv2D(filters=64, kernel_size=(kernel_size, kernel_size), padding='same')(x)
        x = layers.concatenate([x, y])

    # Classification layer
    outputs = layers.Dense(units=10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
model.evaluate(x_test, y_test)