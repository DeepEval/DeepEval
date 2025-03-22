from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose, AveragePooling2D, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Input shape
input_shape = (32, 32, 3)

# Function to create the model
def dl_model(input_shape):
    # Define the input layer
    inputs = Input(shape=input_shape)

    # Branch 1: Local feature extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x1 = MaxPooling2D((2, 2), padding='same')(x)

    # Branch 2 and 3: Downsampling and upsampling
    x2 = AveragePooling2D((2, 2), padding='same')(inputs)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x2 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(x2)

    x3 = AveragePooling2D((2, 2), padding='same')(inputs)
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(x3)
    x3 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(x3)

    # Concatenate the features from all branches
    x = Concatenate()([x1, x2, x3])

    # Refine the features with another 1x1 convolution
    x = Conv2D(128, (1, 1), activation='relu')(x)

    # Fully connected layer for classification
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = dl_model(input_shape)

# Display the model summary
model.summary()

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)