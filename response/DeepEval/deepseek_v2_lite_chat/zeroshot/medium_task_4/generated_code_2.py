import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.datasets import cifar10
from keras.utils import to_categorical


def dl_model():
    # Load and preprocess CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the paths for both pathways
    input_path1 = Input(shape=(32, 32, 3))
    input_path2 = Input(shape=(32, 32, 3))

    # Path 1: Two convolution blocks followed by average pooling
    x = Conv2D(32, (3, 3), activation='relu')(input_path1)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    avg_pool = Flatten()(x)

    # Path 2: Single convolution layer
    y = Conv2D(64, (3, 3), activation='relu')(input_path2)
    flatten_path2 = Flatten()(y)

    # Combine features from both pathways
    concat = Concatenate()([avg_pool, flatten_path2])

    # Fully connected layer to map to class probabilities
    output = Dense(10, activation='softmax')(concat)

    # Define the model
    model = Model(inputs=[input_path1, input_path2], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model