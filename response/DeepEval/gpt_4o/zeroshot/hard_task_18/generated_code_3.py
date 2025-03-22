import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # First block - feature extraction
    # Two 3x3 Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Average pooling layer
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Combine input with the output of the first block using addition
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same')(inputs)
    x = Add()([x, residual])

    # Second block - channel attention
    # Global average pooling
    gap = GlobalAveragePooling2D()(x)

    # Channel weights through two fully connected layers
    channel_weights = Dense(32 // 2, activation='relu')(gap)  # Reduce dimensions
    channel_weights = Dense(32, activation='sigmoid')(channel_weights)  # Back to original dimensions

    # Reshape channel weights
    channel_weights = Reshape((1, 1, 32))(channel_weights)

    # Multiply input by channel weights
    x = Multiply()([x, channel_weights])

    # Flatten and fully connected layer for classification
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# You can train the model using model.fit(x_train, y_train, ...)