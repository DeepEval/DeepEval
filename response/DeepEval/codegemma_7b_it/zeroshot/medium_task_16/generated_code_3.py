import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Split the input into three groups along the channel dimension
    input_tensor = keras.Input(shape=(32, 32, 3))
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_tensor)

    # Apply 1x1 convolutions to each group independently
    conv_outputs = []
    for i in range(3):
        x_i = layers.Conv2D(32 // 3, (1, 1), padding='same')(x[i])
        conv_outputs.append(x_i)

    # Concatenate the three groups of feature maps along the channel dimension
    merged = layers.Concatenate(axis=3)(conv_outputs)

    # Downsample each group of feature maps
    downsampled_outputs = []
    for i in range(3):
        x_i = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(merged[:, :, :, i])
        downsampled_outputs.append(x_i)

    # Concatenate the downsampled feature maps along the channel dimension
    downsampled = layers.Concatenate(axis=3)(downsampled_outputs)

    # Flatten the concatenated feature maps and pass through two fully connected layers
    flattened = layers.Flatten()(downsampled)
    outputs = [
        layers.Dense(256, activation='relu')(flattened),
        layers.Dense(10, activation='softmax')(flattened)
    ]

    # Create the model
    model = keras.Model(inputs=input_tensor, outputs=outputs)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)