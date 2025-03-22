from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Conv2D, Dropout, concatenate, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First Block with Average Pooling
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    # Flatten the outputs from the pooling layers
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    # Concatenate the flattened layers
    concatenated = concatenate([flat1, flat2, flat3])

    # Fully connected layer followed by reshape
    fc = Dense(128, activation='relu')(concatenated)
    reshaped = Reshape((4, 4, 8))(fc)  # Reshape to a 4D tensor for the next block

    # Second Block with parallel paths
    # Path 1
    path1 = Conv2D(8, (1, 1), activation='relu', padding='same')(reshaped)
    path1 = Dropout(0.2)(path1)

    # Path 2
    path2 = Conv2D(8, (1, 1), activation='relu', padding='same')(reshaped)
    path2 = Conv2D(8, (3, 3), activation='relu', padding='same')(path2)
    path2 = Conv2D(8, (3, 3), activation='relu', padding='same')(path2)
    path2 = Dropout(0.2)(path2)

    # Path 3
    path3 = Conv2D(8, (1, 1), activation='relu', padding='same')(reshaped)
    path3 = Conv2D(8, (3, 3), activation='relu', padding='same')(path3)
    path3 = Dropout(0.2)(path3)

    # Path 4
    path4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(reshaped)
    path4 = Conv2D(8, (1, 1), activation='relu', padding='same')(path4)
    path4 = Dropout(0.2)(path4)

    # Concatenate all paths
    concatenated_paths = concatenate([path1, path2, path3, path4], axis=-1)

    # Fully connected layers for classification
    flat = Flatten()(concatenated_paths)
    fc1 = Dense(128, activation='relu')(flat)
    output_layer = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()