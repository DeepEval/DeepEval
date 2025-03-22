from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10

def identity_block(X, filters):
    # Save the input X
    X_shortcut = X

    # First component of main path
    X = layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    # Second component of main path
    X = layers.Conv2D(filters=filters[1], kernel_size=(3, 3), strides=(1, 1), padding='same')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    # Third component of main path
    X = layers.Conv2D(filters=filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)

    # Shortcut path
    X_shortcut = layers.Conv2D(filters=filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid')(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis=3)(X_shortcut)

    # Add main path and shortcut path
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X

def convolutional_block(X, filters, s=1):
    # Save the input X
    X_shortcut = X

    # First component of main path
    X = layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(s, s), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    # Second component of main path
    X = layers.Conv2D(filters=filters[1], kernel_size=(3, 3), strides=(1, 1), padding='same')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    # Third component of main path
    X = layers.Conv2D(filters=filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)

    # Shortcut path
    X_shortcut = layers.Conv2D(filters=filters[2], kernel_size=(1, 1), strides=(s, s), padding='valid')(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis=3)(X_shortcut)

    # Add main path and shortcut path
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X

def dl_model():
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Preprocess the input data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Create the model
    X_input = layers.Input(shape=(32, 32, 3))
    X = layers.ZeroPadding2D((3, 3))(X_input)
    X = layers.Conv2D(64, (3, 3), strides=(1, 1))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    # Stage 1
    X = convolutional_block(X, filters=[64, 64, 256], s=1)
    X = identity_block(X, filters=[64, 64, 256])

    # Stage 2
    X = convolutional_block(X, filters=[128, 128, 512], s=2)
    X = identity_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])

    # Stage 3
    X = convolutional_block(X, filters=[512, 512, 2048], s=2)
    X = identity_block(X, filters=[512, 512, 2048])
    X = identity_block(X, filters=[512, 512, 2048])

    # Average pooling and fully connected layer
    X = layers.AveragePooling2D((2, 2))(X)
    X = layers.Flatten()(X)
    X = layers.Dense(units=10, activation='softmax')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=64)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)

    # Return the model
    return model