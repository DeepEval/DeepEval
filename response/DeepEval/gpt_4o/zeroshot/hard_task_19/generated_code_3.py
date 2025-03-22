import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Multiply, Add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 channels
    num_classes = 10

    inputs = Input(shape=input_shape)

    # Main path: three 3x3 convolutional layers followed by a max pooling layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Branch path: global average pooling followed by two fully connected layers
    y = GlobalAveragePooling2D()(inputs)
    y = Dense(64, activation='relu')(y)
    y = Dense(128, activation='sigmoid')(y)  # Output channel weights
    y = tf.keras.layers.Reshape((1, 1, 128))(y)  # Reshape to match the dimensions for multiplication

    # Multiply channel weights with the input
    scaled_inputs = Multiply()([x, y])

    # Combine outputs from both paths
    combined = Add()([x, scaled_inputs])

    # Two additional fully connected layers for classification
    z = Flatten()(combined)
    z = Dense(256, activation='relu')(z)
    outputs = Dense(num_classes, activation='softmax')(z)

    # Build the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Display the model architecture
model.summary()