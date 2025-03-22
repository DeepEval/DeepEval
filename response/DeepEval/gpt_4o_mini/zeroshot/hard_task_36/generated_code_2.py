import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(28, 28, 1))

    # Main pathway
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # Branch pathway
    branch = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)

    # Fusion of pathways
    fused = layers.add([x, branch])

    # Global Average Pooling, Flatten and Fully Connected Layer
    x = layers.GlobalAveragePooling2D()(fused)
    x = layers.Flatten()(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    # Construct model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Optionally, to compile and summarize the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Load and prepare the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)