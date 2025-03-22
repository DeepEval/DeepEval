from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 Convolution
    conv_1x1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # 3x3 Convolution
    conv_3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    # 5x5 Convolution
    conv_5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)

    # 3x3 Max Pooling
    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    conv_pool = Conv2D(32, (1, 1), activation='relu', padding='same')(max_pool)

    # Concatenate all the features
    concatenated = Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5, conv_pool])

    # Flatten the concatenated features
    flatten = Flatten()(concatenated)

    # Fully connected layers
    fc1 = Dense(256, activation='relu')(flatten)
    dropout1 = Dropout(0.5)(fc1)
    fc2 = Dense(128, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(fc2)

    # Output layer
    output_layer = Dense(10, activation='softmax')(dropout2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Load CIFAR-10 data for testing the model creation
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()