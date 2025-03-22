from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Reshape, Multiply, Flatten, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    input_shape = x_train.shape[1:]
    inputs = Input(shape=input_shape)

    # First Block
    # Initial Convolutional Layer
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2))(conv2)

    # Add the input to the output of the main path
    add1 = Add()([inputs, avg_pool1])

    # Second Block
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(add1)

    # Fully Connected Layers for Channel Weights
    dense1 = Dense(32, activation='relu')(gap)
    dense2 = Dense(32, activation='sigmoid')(dense1)

    # Reshape to multiply with input
    channel_weights = Reshape((1, 1, 32))(dense2)
    scaled_features = Multiply()([add1, channel_weights])

    # Flatten and Final Dense Layer for Classification
    flat = Flatten()(scaled_features)
    outputs = Dense(10, activation='softmax')(flat)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Usage example: instantiate and train the model
model = dl_model()