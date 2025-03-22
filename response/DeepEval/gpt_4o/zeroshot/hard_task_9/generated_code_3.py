from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Concatenate, Add, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    num_classes = 10  # CIFAR-10 has 10 classes

    inputs = Input(shape=input_shape)

    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)

    # Branch 3: 1x1 Convolution followed by two 3x3 Convolutions
    branch3 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)

    # Concatenate the outputs of the branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # 1x1 Convolution to adjust the output dimensions
    adjust_channels = Conv2D(3, (1, 1), activation='relu', padding='same')(concatenated)

    # Add the adjusted output with the input
    main_path_output = Add()([inputs, adjust_channels])

    # Global Average Pooling
    x = GlobalAveragePooling2D()(main_path_output)

    # Fully connected layers for classification
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
if __name__ == "__main__":
    # Load and preprocess CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    model = dl_model()
    model.summary()
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)