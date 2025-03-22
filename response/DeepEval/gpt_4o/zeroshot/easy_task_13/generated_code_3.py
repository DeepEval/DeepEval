from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    input_shape = x_train.shape[1:]
    inputs = Input(shape=input_shape)

    # First 1x1 convolutional layer with dropout
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    x = Dropout(0.2)(x)

    # Second 1x1 convolutional layer with dropout
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)

    # 3x1 convolutional layer with dropout
    x = Conv2D(filters=32, kernel_size=(3, 1), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)

    # 1x3 convolutional layer with dropout
    x = Conv2D(filters=32, kernel_size=(1, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)

    # 1x1 convolutional layer to restore the number of input channels
    x = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)

    # Add processed features with the original input
    x = Add()([x, inputs])

    # Flatten layer
    x = Flatten()(x)

    # Fully connected layer for classification
    outputs = Dense(units=10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model