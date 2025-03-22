import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the main path of the model
def main_path(inputs):
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (5, 5), padding='same', activation='relu')(x)
    x1, x2, x3 = tf.split(x, 3, axis=-1)
    x2_pool = MaxPooling2D(pool_size=(2, 2))(x2)
    x3_pool = MaxPooling2D(pool_size=(2, 2))(x3)
    x = tf.concat([x1, x2_pool, x3_pool], axis=-1)
    return x

# Define the branch path of the model
def branch_path(inputs):
    x = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)
    return x

# Combine the main and branch paths
def fused_features(inputs):
    x = main_path(inputs)
    x = branch_path(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    return x

# Construct the model
def dl_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = main_path(inputs)
    x = branch_path(x)
    outputs = fused_features(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile the model
model = dl_model((32, 32, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Export the model to HDF5 format
model.save('cifar10_model.h5')