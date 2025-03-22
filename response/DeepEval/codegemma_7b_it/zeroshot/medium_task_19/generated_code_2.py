import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Preprocess data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Define input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution for dimensionality reduction
    branch_1 = layers.Conv2D(64, (1, 1), padding='same')(input_layer)

    # Branch 2: 1x1 + 3x3 convolution for feature extraction
    branch_2 = layers.Conv2D(64, (1, 1), padding='same')(input_layer)
    branch_2 = layers.Conv2D(128, (3, 3), padding='same')(branch_2)

    # Branch 3: 1x1 + 5x5 convolution for capturing larger spatial information
    branch_3 = layers.Conv2D(64, (1, 1), padding='same')(input_layer)
    branch_3 = layers.Conv2D(256, (5, 5), padding='same')(branch_3)

    # Branch 4: 3x3 max pooling + 1x1 convolution for downsampling and further processing
    branch_4 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(input_layer)
    branch_4 = layers.Conv2D(128, (1, 1), padding='same')(branch_4)

    # Concatenate outputs of all branches
    concat_layer = layers.concatenate([branch_1, branch_2, branch_3, branch_4])

    # Flatten features
    flatten_layer = layers.Flatten()(concat_layer)

    # Fully connected layers for feature combination and classification
    dense_layer1 = layers.Dense(512, activation='relu')(flatten_layer)
    dense_layer2 = layers.Dense(10, activation='softmax')(dense_layer1)

    # Construct model
    model = keras.Model(inputs=input_layer, outputs=dense_layer2)

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model (optional)
    # model.fit(x_train, y_train, epochs=10)

    return model

# Construct model
model = dl_model()

# Print model summary
model.summary()