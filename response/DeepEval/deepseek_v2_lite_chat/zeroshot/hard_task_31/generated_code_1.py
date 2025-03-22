import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Add, Concatenate, Input, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the input shape
input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels

def dl_model():
    # First block
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

    # Main path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input1)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.2)(x)

    # Branch path
    y = input2

    # Add the paths
    z1 = x
    z2 = y
    z = Add()([z1, z2])

    # Second block
    # Split branch path to three groups
    split1 = tf.split(z, 3, axis=-1)
    split2 = tf.split(y, 3, axis=-1)

    # Separate convolutions for each group
    x1 = Conv2D(64, (1, 1), activation='relu')(split1[0])
    x2 = Conv2D(64, (3, 3), activation='relu')(split1[1])
    x3 = Conv2D(64, (5, 5), activation='relu')(split1[2])

    y1 = Conv2D(64, (1, 1), activation='relu')(split2[0])
    y2 = Conv2D(64, (3, 3), activation='relu')(split2[1])
    y3 = Conv2D(64, (5, 5), activation='relu')(split2[2])

    # Dropout layers
    x1 = Dropout(0.2)(x1)
    x2 = Dropout(0.2)(x2)
    x3 = Dropout(0.2)(x3)
    y1 = Dropout(0.2)(y1)
    y2 = Dropout(0.2)(y2)
    y3 = Dropout(0.2)(y3)

    # Concatenate outputs from the three groups
    z1 = Concatenate(axis=-1)([x1, x2, x3])
    z2 = Concatenate(axis=-1)([y1, y2, y3])

    # Flatten and fully connected layers
    z = Flatten()(z1)
    z = Dense(256, activation='relu')(z)
    z = Dropout(0.5)(z)

    output = Dense(10, activation='softmax')(z)

    # Define the model
    model = Model(inputs=[input1, input2], outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build and train the model
model = dl_model()
model.fit([x_train, x_train], y_train, validation_data=([x_test, x_test], y_test), epochs=10, batch_size=64)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate([x_test, x_test], y_test, verbose=2)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)