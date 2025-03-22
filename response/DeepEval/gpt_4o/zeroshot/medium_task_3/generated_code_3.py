from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 channel
    num_classes = 10  # 10 categories for classification (digits 0-9)

    # Input layer
    inputs = Input(shape=input_shape)

    # First convolutional block
    x1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)

    # Second convolutional block
    x2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x1)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)

    # Ensure the last convolution output has the same channel dimension as the input
    x2 = Conv2D(filters=input_shape[-1], kernel_size=(1, 1), activation='relu', padding='same')(x2)

    # Adding the output of the second conv block with the original input
    x2 = Add()([x2, inputs])

    # Flatten and fully connected layer for classification
    x = Flatten()(x2)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])