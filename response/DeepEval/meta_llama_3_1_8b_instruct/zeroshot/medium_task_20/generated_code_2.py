from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

def dl_model():
    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize pixel values between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Convert class labels to categorical labels
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first path with a 1x1 convolution
    x1 = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, (1, 1), activation='relu')(x1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    flatten1 = layers.Flatten()(pool1)

    # Define the second path with two 3x3 convolutions stacked after a 1x1 convolution
    x2 = layers.Input(shape=input_shape)
    conv2 = layers.Conv2D(32, (1, 1), activation='relu')(x2)
    conv3 = layers.Conv2D(32, (3, 3), activation='relu')(conv2)
    conv4 = layers.Conv2D(32, (3, 3), activation='relu')(conv3)
    pool2 = layers.MaxPooling2D((2, 2))(conv4)
    flatten2 = layers.Flatten()(pool2)

    # Define the third path with a single 3x3 convolution following a 1x1 convolution
    x3 = layers.Input(shape=input_shape)
    conv5 = layers.Conv2D(32, (1, 1), activation='relu')(x3)
    conv6 = layers.Conv2D(32, (3, 3), activation='relu')(conv5)
    pool3 = layers.MaxPooling2D((2, 2))(conv6)
    flatten3 = layers.Flatten()(pool3)

    # Define the fourth path with max pooling followed by a 1x1 convolution
    x4 = layers.Input(shape=input_shape)
    conv7 = layers.Conv2D(32, (3, 3), activation='relu')(x4)
    pool4 = layers.MaxPooling2D((2, 2))(conv7)
    conv8 = layers.Conv2D(32, (1, 1), activation='relu')(pool4)
    flatten4 = layers.Flatten()(conv8)

    # Concatenate the outputs from all paths
    concatenated = layers.Concatenate()([flatten1, flatten2, flatten3, flatten4])

    # Add a dense layer with 128 units
    dense = layers.Dense(128, activation='relu')(concatenated)

    # Define the output layer with softmax activation
    output = layers.Dense(10, activation='softmax')(dense)

    # Create the model
    model = Model(inputs=[x1, x2, x3, x4], outputs=output)

    return model