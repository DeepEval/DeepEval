from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Define the model
    input_layer = layers.Input(shape=(32, 32, 3))
    path1 = layers.Conv2D(64, (3, 3), activation='relu')(input_layer)
    path1 = layers.AveragePooling2D((2, 2))(path1)
    path1 = layers.Conv2D(128, (3, 3), activation='relu')(path1)
    path1 = layers.AveragePooling2D((2, 2))(path1)
    path2 = layers.Conv2D(128, (3, 3), activation='relu')(input_layer)
    path2 = layers.AveragePooling2D((2, 2))(path2)
    concat = layers.concatenate([path1, path2])
    flatten = layers.Flatten()(concat)
    output_layer = layers.Dense(10, activation='softmax')(flatten)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10)

    # Evaluate the model
    model.evaluate(x_test, y_test)

    return model