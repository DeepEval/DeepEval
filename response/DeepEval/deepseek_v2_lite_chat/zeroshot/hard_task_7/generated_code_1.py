import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda, Flatten, Dense, Reshape, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Parameters
input_shape = (28, 28, 1)
num_classes = 10

def dl_model():
    # Block 1
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same')(inputs)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=1, padding='same')(x)
    
    split = Lambda(lambda x: tf.split(x, 2, axis=-1))(x)
    x1 = split[0]
    x2 = split[1]
    
    x1 = tf.keras.layers.Conv2D(32, kernel_size=1, padding='same')(x1)
    x1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x1)
    x1 = tf.keras.layers.Conv2D(32, kernel_size=1, padding='same')(x1)
    
    x2 = tf.keras.layers.Conv2D(32, kernel_size=1, padding='same')(x2)
    
    # Block 2
    x1 = tf.keras.layers.Flatten()(x1)
    x2 = tf.keras.layers.Flatten()(x2)
    
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Permute((2, 3, 1))(x)
    x = tf.keras.layers.Reshape((-1, x.shape[1] // 2 * x.shape[2]))(x)
    
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

# Create the model
model = dl_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)