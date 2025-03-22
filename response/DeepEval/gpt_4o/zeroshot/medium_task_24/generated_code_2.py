from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image size
    num_classes = 10  # CIFAR-10 has 10 classes

    # Input Layer
    inputs = Input(shape=input_shape)

    # Branch 1: 1x1 convolution, 3x3 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = Dropout(0.3)(branch1)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    branch2 = Conv2D(32, (1, 7), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(32, (7, 1), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Dropout(0.3)(branch2)

    # Branch 3: Max pooling
    branch3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch3 = Dropout(0.3)(branch3)

    # Concatenate all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten and add fully connected layers
    x = Flatten()(concatenated)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Create the model
model = dl_model()

# Print model summary
model.summary()

# You can now train the model using model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))