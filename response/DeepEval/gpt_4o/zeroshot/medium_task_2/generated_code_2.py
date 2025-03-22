from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Main path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Branch path
    y = Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)

    # Combine paths
    combined = concatenate([x, y])

    # Flatten the combined features
    flattened = Flatten()(combined)

    # Fully connected layers
    fc1 = Dense(256, activation='relu')(flattened)
    output_layer = Dense(10, activation='softmax')(fc1)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()