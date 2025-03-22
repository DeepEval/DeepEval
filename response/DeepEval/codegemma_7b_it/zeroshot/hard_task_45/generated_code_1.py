from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model
def dl_model():

    # Input layer
    input_img = layers.Input(shape=(32, 32, 3))

    # First block
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_img)
    x0 = layers.Lambda(lambda x: tf.keras.backend.depthwise_conv2d(x, kernel_size=1, padding='same'))(x[0])
    x0 = layers.Conv2D(64, (1, 1), padding='same')(x0)
    x1 = layers.Lambda(lambda x: tf.keras.backend.depthwise_conv2d(x, kernel_size=3, padding='same'))(x[1])
    x1 = layers.Conv2D(64, (1, 1), padding='same')(x1)
    x2 = layers.Lambda(lambda x: tf.keras.backend.depthwise_conv2d(x, kernel_size=5, padding='same'))(x[2])
    x2 = layers.Conv2D(64, (1, 1), padding='same')(x2)
    x = layers.Concatenate()([x0, x1, x2])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second block
    branches = []
    branches.append(layers.Conv2D(64, (1, 1), padding='same')(x))
    branches.append(layers.Conv2D(64, (1, 1), padding='same')(layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)))
    branches.append(layers.Conv2D(64, (3, 3), padding='same')(x))
    branches.append(layers.Conv2D(64, (3, 3), padding='same')(layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)))
    branches.append(layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x))
    branches.append(layers.Conv2D(64, (1, 1), padding='same')(branches[4]))
    x = layers.Concatenate()(branches)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Output layer
    x = layers.Flatten()(x)
    output = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=input_img, outputs=output)

    return model

# Compile and train the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])