from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Image data generator for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # Build the VGG16 model to extract features from the images
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Pathways for the model
    input_path1 = Input(shape=(7, 7, 512))
    input_path2 = Input(shape=(3, 3, 512))

    # Path 1: Two blocks of convolution, average pooling, and flattening
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_path1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Flatten()(x1)

    # Path 2: Single convolution layer and flattening
    x2 = Conv2D(64, (1, 1), activation='relu', padding='same')(input_path2)
    x2 = Flatten()(x2)

    # Combine the outputs of both pathways
    combined = Add()([x1, x2])

    # Fully connected layer to map the combined features to probabilities
    output = Dense(10, activation='softmax')(combined)

    # Assemble the model
    model = Model(inputs=[input_path1, input_path2], outputs=output)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model