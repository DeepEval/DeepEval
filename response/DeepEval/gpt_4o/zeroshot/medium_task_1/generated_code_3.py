from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset (optional here, but useful for context)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    # Second convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # Max-pooling layer
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Adding the extracted features to the input
    x = Add()([x, input_layer])

    # Flatten the output for the fully connected layers
    x = Flatten()(x)

    # First fully connected layer
    x = Dense(64, activation='relu')(x)
    # Second fully connected layer
    output_layer = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model