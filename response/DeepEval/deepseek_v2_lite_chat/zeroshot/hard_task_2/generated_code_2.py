import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
    x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

    # Split input into three groups along the channel dimension
    split_dim = 1
    x_train_split1 = tf.split(x_train, 3, axis=split_dim)
    x_train_split2 = tf.split(x_train, 3, axis=split_dim)
    x_train_split3 = tf.split(x_train, 3, axis=split_dim)

    # Define the main path of the model
    inputs = Input(shape=(32, 32, 3))
    x = inputs

    # First 1x1 convolution
    x = Conv2D(32, (1, 1), activation='relu')(x)
    # First 3x3 convolution
    x = Conv2D(64, (3, 3), activation='relu')(x)
    # Second 1x1 convolution
    x = Conv2D(64, (1, 1), activation='relu')(x)

    # Second path
    x1 = x_train_split1
    for _ in range(3):
        x1 = Conv2D(64, (3, 3), activation='relu')(x1)
        x1 = Conv2D(64, (1, 1), activation='relu')(x1)

    x2 = x_train_split2
    for _ in range(3):
        x2 = Conv2D(64, (3, 3), activation='relu')(x2)
        x2 = Conv2D(64, (1, 1), activation='relu')(x2)

    x3 = x_train_split3
    for _ in range(3):
        x3 = Conv2D(64, (3, 3), activation='relu')(x3)
        x3 = Conv2D(64, (1, 1), activation='relu')(x3)

    # Combine outputs
    combined = Concatenate()([x, x1, x2, x3])

    # Fusion with original input
    x = Concatenate()([x, combined])

    # Flatten and feed into fully connected layer
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=Adam(),
                  metrics=['accuracy'])

    return model

# Instantiate and return the model
model = dl_model()