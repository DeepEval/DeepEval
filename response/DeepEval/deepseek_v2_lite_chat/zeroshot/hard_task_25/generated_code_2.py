import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dense
from tensorflow.keras.optimizers import Adam


def dl_model():
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Image dimensions
    img_shape = (32, 32, 3)

    # Input layers
    input_main = Input(shape=img_shape)
    input_branch = Input(shape=(10,))  # Match the number of channels

    # Main path layers
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_main)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Branch path layers
    x_branch = Conv2D(32, (1, 1), activation='relu')(input_main)  # Match main path channels
    x_branch_1 = MaxPooling2D((3, 3), padding='same')(x)
    x_branch_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_branch_1)
    x_branch = UpSampling2D((2, 2))(x_branch_2)
    x_branch = Concatenate()([x, x_branch])

    # Final layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x_branch)
    output = Dense(10, activation='softmax')(x)

    # Model
    model = Model(inputs=[input_main, input_branch], outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Return the model
    return model

# Example usage
model = dl_model()
model.fit([x_train, x_train], y_train, epochs=10, validation_data=([x_test, x_test], y_test))