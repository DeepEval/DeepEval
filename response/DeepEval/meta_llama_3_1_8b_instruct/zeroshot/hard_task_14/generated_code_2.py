# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize input data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define input shape
    input_shape = (32, 32, 3)

    # Define main path
    main_input = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(main_input)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Define branch path
    branch_input = keras.Input(shape=input_shape)
    x = layers.Conv2D(128, (3, 3), activation='relu')(branch_input)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Combine main and branch paths
    x = keras.layers.Concatenate()([x, x])
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Define model
    model = keras.Model(inputs=[main_input, branch_input], outputs=x)

    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create and compile the model
model = dl_model()
print(model.summary())