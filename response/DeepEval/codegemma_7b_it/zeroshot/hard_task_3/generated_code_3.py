import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():

    # Input layer for the CIFAR-10 dataset
    inputs = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    x1, x2, x3 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(inputs)

    # Main pathway
    x1 = layers.Conv2D(32, (1, 1), padding='same')(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(64, (3, 3), padding='same')(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Dropout(0.5)(x1)

    x2 = layers.Conv2D(32, (1, 1), padding='same')(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.Conv2D(64, (3, 3), padding='same')(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.Dropout(0.5)(x2)

    x3 = layers.Conv2D(32, (1, 1), padding='same')(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = layers.Conv2D(64, (3, 3), padding='same')(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = layers.Dropout(0.5)(x3)

    # Concatenate the outputs from the three groups
    x = layers.concatenate([x1, x2, x3])

    # Branch pathway
    y = layers.Conv2D(64, (1, 1), padding='same')(inputs)
    y = layers.Activation('relu')(y)

    # Combine the outputs from both pathways
    combined = layers.add([x, y])
    combined = layers.Activation('relu')(combined)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(combined)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])