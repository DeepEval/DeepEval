from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path: Two consecutive 3x3 convolutional layers followed by max pooling
    main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    main_path = Conv2D(64, (3, 3), activation='relu', padding='same')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)

    # Branch path: Single 5x5 convolutional layer
    branch_path = Conv2D(64, (5, 5), activation='relu', padding='same')(input_layer)

    # Combine features from both paths
    combined = concatenate([main_path, branch_path])

    # Flatten the combined features
    flat = Flatten()(combined)

    # Fully connected layers to map to probability distribution
    fc1 = Dense(128, activation='relu')(flat)
    output_layer = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
model = dl_model()
model.summary()