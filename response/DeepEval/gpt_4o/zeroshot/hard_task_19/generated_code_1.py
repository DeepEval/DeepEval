from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Multiply, Add, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32x3
    inputs = Input(shape=input_shape)

    # Main path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    main_path_output = Flatten()(x)

    # Branch path
    branch = GlobalAveragePooling2D()(inputs)
    branch = Dense(64, activation='relu')(branch)
    branch = Dense(32, activation='sigmoid')(branch)
    branch = Reshape((1, 1, 32))(branch)

    # Multiply the input by the channel weights
    scaled_inputs = Multiply()([inputs, branch])

    # Combine both paths
    combined = Add()([x, scaled_inputs])
    combined = Flatten()(combined)

    # Fully connected layers for classification
    combined = Dense(256, activation='relu')(combined)
    combined = Dense(128, activation='relu')(combined)
    outputs = Dense(10, activation='softmax')(combined)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# To test the model with CIFAR-10 data
if __name__ == "__main__":
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Instantiate and compile the model
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()