from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Reshape, Concatenate, Conv2D, MaxPooling2D, Dropout, Lambda
from keras.models import Model

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Input shape
    input_shape = (28, 28, 1)

    # Reshape data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Add a channels dimension
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # First block: Pooling and flattening
    input_layer = Input(shape=input_shape)
    x = input_layer

    # Pooling layers
    x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x)  # 1x1
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)  # 2x2
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(x)  # 4x4

    # Flatten and concatenate
    x = Flatten()(x)
    x = Concatenate()([x, x])  # Concatenate all pooling outputs

    # Second block: Convolution and dropout
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(x)  # 1x1 conv
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)  # 3x3 conv
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)  # 3x3 conv
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)  # average pooling
    x = Conv2D(128, kernel_size=(1, 1), activation='relu')(x)  # 1x1 conv

    # Dropout
    x = Dropout(0.5)(x)

    # Output layer
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Build and train the model
model = dl_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)